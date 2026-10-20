// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Projection on the partition-index axis.
//!
//! The scheduler restricts a stage plan to one task's partition slice by
//! rewriting its leaves (see the task builder). That works for leaves it
//! knows how to rebuild with fewer partitions — shuffle readers, file and
//! in-memory scans. A leaf it knows nothing about, such as a user-defined
//! scan, would be shipped at its full partition count, and since the
//! executor runs every partition of the plan it receives, every task of the
//! stage would produce the leaf's whole output.
//!
//! This operator closes that gap using nothing but the `ExecutionPlan`
//! contract: output partition `j` is `input.execute(partitions[j])`. The
//! input is left untouched, so it is serialized by whichever codec already
//! handles it and only the selected indices travel alongside.

use std::fmt::{self, Debug, Formatter};
use std::sync::Arc;

use datafusion::arrow::datatypes::SchemaRef;
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::common::{Result, Statistics, internal_err};
use datafusion::execution::TaskContext;
use datafusion::physical_expr::{Distribution, OrderingRequirements, PhysicalExpr};
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, Partitioning,
    PlanProperties, SendableRecordBatchStream, StatisticsArgs, statistics::ChildStats,
};

/// Exposes a subset of its input's partitions, re-indexed so output
/// partition `j` is input partition `partitions[j]`.
pub struct PartitionSelectExec {
    input: Arc<dyn ExecutionPlan>,
    partitions: Vec<usize>,
    properties: Arc<PlanProperties>,
}

impl PartitionSelectExec {
    /// Select `partitions` of `input`, in that order.
    ///
    /// Fails if any index is out of range for the input's partition count.
    /// An empty selection is valid and yields a 0-partition plan.
    pub fn try_new(
        input: Arc<dyn ExecutionPlan>,
        partitions: Vec<usize>,
    ) -> Result<Self> {
        let input_count = input.output_partitioning().partition_count();
        if let Some(&p) = partitions.iter().find(|&&p| p >= input_count) {
            return internal_err!(
                "PartitionSelectExec: partition {p} out of bounds for an input \
                 with {input_count} partitions"
            );
        }
        let partitioning =
            select_partitioning(input.output_partitioning(), partitions.len());
        let properties = Arc::new(PlanProperties::new(
            input.equivalence_properties().clone(),
            partitioning,
            input.pipeline_behavior(),
            input.boundedness(),
        ));
        Ok(Self {
            input,
            partitions,
            properties,
        })
    }

    /// The input this operator selects from.
    pub fn input(&self) -> &Arc<dyn ExecutionPlan> {
        &self.input
    }

    /// The selected input partitions. `partitions()[j]` is served as output
    /// partition `j`.
    pub fn partitions(&self) -> &[usize] {
        &self.partitions
    }
}

/// Shrink `partitioning` to `count` partitions, keeping its kind.
///
/// `Range` becomes `UnknownPartitioning`: a selection need not be contiguous,
/// so the split points of the kept partitions can't in general describe it.
/// Claiming nothing is weaker than the input's claim but still true.
fn select_partitioning(partitioning: &Partitioning, count: usize) -> Partitioning {
    match partitioning {
        Partitioning::Hash(exprs, _) => Partitioning::Hash(exprs.clone(), count),
        Partitioning::RoundRobinBatch(_) => Partitioning::RoundRobinBatch(count),
        Partitioning::Range(_) | Partitioning::UnknownPartitioning(_) => {
            Partitioning::UnknownPartitioning(count)
        }
    }
}

impl Debug for PartitionSelectExec {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("PartitionSelectExec")
            .field("partitions", &self.partitions)
            .finish()
    }
}

impl DisplayAs for PartitionSelectExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut Formatter<'_>) -> fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(f, "PartitionSelectExec: partitions={:?}", self.partitions)
            }
            DisplayFormatType::TreeRender => {
                write!(f, "partitions={:?}", self.partitions)
            }
        }
    }
}

impl ExecutionPlan for PartitionSelectExec {
    fn name(&self) -> &str {
        "PartitionSelectExec"
    }

    fn schema(&self) -> SchemaRef {
        self.input.schema()
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    /// Owns no expressions — selection is by partition index only.
    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let [input] = children.as_slice() else {
            return internal_err!(
                "PartitionSelectExec expects exactly one child, got {}",
                children.len()
            );
        };
        Ok(Arc::new(PartitionSelectExec::try_new(
            input.clone(),
            self.partitions.clone(),
        )?))
    }

    fn required_input_distribution(&self) -> Vec<Distribution> {
        vec![Distribution::UnspecifiedDistribution]
    }

    fn required_input_ordering(&self) -> Vec<Option<OrderingRequirements>> {
        vec![None]
    }

    /// Each output partition is an input partition streamed verbatim.
    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true]
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![false]
    }

    /// A single partition's statistics are its input partition's. The
    /// overall statistics would have to be merged from the selected
    /// partitions only, so they are left unknown.
    fn statistics_from_inputs(
        &self,
        input_stats: &[Arc<Statistics>],
        args: &StatisticsArgs,
    ) -> Result<Arc<Statistics>> {
        match args.partition() {
            Some(_) => Ok(Arc::clone(&input_stats[0])),
            None => Ok(Arc::new(Statistics::new_unknown(&self.schema()))),
        }
    }

    fn child_stats_requests(&self, partition: Option<usize>) -> Vec<ChildStats> {
        match partition.and_then(|j| self.partitions.get(j)) {
            Some(&p) => vec![ChildStats::At(Some(p))],
            None => vec![ChildStats::Skip],
        }
    }

    fn execute(
        &self,
        partition: usize,
        ctx: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let Some(&input_partition) = self.partitions.get(partition) else {
            return internal_err!(
                "PartitionSelectExec: partition {} out of bounds ({} selected)",
                partition,
                self.partitions.len()
            );
        };
        self.input.execute(input_partition, ctx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::array::{Int64Array, RecordBatch};
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use datafusion::datasource::memory::MemorySourceConfig;
    use datafusion::datasource::source::DataSourceExec;
    use datafusion::physical_expr::expressions::Column;
    use datafusion::physical_plan::repartition::RepartitionExec;
    use datafusion::prelude::SessionContext;
    use futures::TryStreamExt;

    fn one_col_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]))
    }

    /// Memory source with `partitions` partitions, partition `k` holding the
    /// single value `k`.
    fn partitioned_source(partitions: usize) -> Arc<dyn ExecutionPlan> {
        let schema = one_col_schema();
        let per_partition: Vec<Vec<RecordBatch>> = (0..partitions as i64)
            .map(|k| {
                let array = Int64Array::from(vec![k]);
                vec![RecordBatch::try_new(schema.clone(), vec![Arc::new(array)]).unwrap()]
            })
            .collect();
        let src =
            MemorySourceConfig::try_new(&per_partition, schema, None).expect("mem src");
        Arc::new(DataSourceExec::new(Arc::new(src)))
    }

    async fn collect(plan: &Arc<dyn ExecutionPlan>, partition: usize) -> Vec<i64> {
        let stream = plan
            .execute(partition, SessionContext::new().task_ctx())
            .unwrap();
        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();
        batches
            .iter()
            .flat_map(|b| {
                b.column(0)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .expect("Int64Array")
                    .values()
                    .to_vec()
            })
            .collect()
    }

    /// Output partition `j` serves input partition `partitions[j]`, in the
    /// requested order rather than the input's.
    #[tokio::test]
    async fn serves_selected_partitions_in_requested_order() {
        let plan: Arc<dyn ExecutionPlan> = Arc::new(
            PartitionSelectExec::try_new(partitioned_source(4), vec![3, 1]).unwrap(),
        );
        assert_eq!(plan.output_partitioning().partition_count(), 2);
        assert_eq!(collect(&plan, 0).await, vec![3]);
        assert_eq!(collect(&plan, 1).await, vec![1]);
        assert!(plan.execute(2, SessionContext::new().task_ctx()).is_err());
    }

    /// An empty selection is a 0-partition plan: a task that polls nothing
    /// from this input.
    #[test]
    fn empty_selection_has_no_partitions() {
        let plan = PartitionSelectExec::try_new(partitioned_source(4), vec![]).unwrap();
        assert_eq!(plan.properties().output_partitioning().partition_count(), 0);
    }

    #[test]
    fn rejects_out_of_range_partition() {
        let err = PartitionSelectExec::try_new(partitioned_source(2), vec![0, 2])
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("partition 2 out of bounds") && err.contains("2 partitions"),
            "unexpected error: {err}"
        );
    }

    /// The partitioning kind survives, only the count shrinks.
    #[test]
    fn keeps_hash_partitioning_kind() {
        let exprs: Vec<Arc<dyn PhysicalExpr>> = vec![Arc::new(Column::new("v", 0))];
        let input: Arc<dyn ExecutionPlan> = Arc::new(
            RepartitionExec::try_new(
                partitioned_source(1),
                Partitioning::Hash(exprs.clone(), 4),
            )
            .unwrap(),
        );
        let plan = PartitionSelectExec::try_new(input, vec![0, 2]).unwrap();
        match plan.properties().output_partitioning() {
            Partitioning::Hash(kept, 2) => assert_eq!(kept.len(), exprs.len()),
            other => panic!("expected Hash(_, 2), got {other}"),
        }
    }
}
