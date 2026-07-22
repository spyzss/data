import test from "node:test";
import assert from "node:assert/strict";

import {
  SemanticCalibrationAdapter,
  buildTimelineModel,
  linkedBoundaryPreview,
  makeBoundaryPayload,
  pendingPresentation,
  renderTimelineMarkup,
} from "./semantic_adapter.js";

const segments = [
  { internal_id: "s1", start_frame: 0, end_frame_exclusive: 51, text_cn: "接近杯子", text_en: "approach" },
  { internal_id: "s2", start_frame: 51, end_frame_exclusive: 123, text_cn: "拿起杯子", text_en: "pick up" },
  { internal_id: "s3", start_frame: 123, end_frame_exclusive: 195, text_cn: "放入托盘", text_en: "place" },
];

const semantic = {
  report_revision: 4,
  report_state: "in_progress",
  timeline: { frame_count: 195, fps: 30, segments },
  pending_edit: null,
};

test("N segments render exactly N-1 internal shared-boundary handles", () => {
  const model = buildTimelineModel(semantic);
  assert.equal(model.handles.length, 2);
  assert.deepEqual(model.handles.map((item) => item.boundary_index), [1, 2]);
  const markup = renderTimelineMarkup(model);
  assert.equal((markup.match(/class="boundary-handle/g) || []).length, 2);
  assert.doesNotMatch(markup, /class="timeline-segment[^>]*draggable=/);
});

test("boundary payload uses the internal half-open frame", () => {
  assert.deepEqual(makeBoundaryPayload({
    boundaryIndex: 1,
    actorSegmentId: "s2",
    frameExclusive: 60,
    expectedRevision: 4,
    leaseToken: "lease-1",
  }), {
    boundary_index: 1,
    actor_segment_id: "s2",
    new_frame_exclusive: 60,
    expected_revision: 4,
    lease_token: "lease-1",
  });
});

test("drag preview changes exactly two adjacent segments", () => {
  const preview = linkedBoundaryPreview(buildTimelineModel(semantic), 1, 60);
  assert.equal(preview.previous.endFrameExclusive, 60);
  assert.equal(preview.following.startFrame, 60);
});

test("pending presentation retains both before and after snapshots", () => {
  const pending = pendingPresentation({
    edit_type: "boundary",
    affected_segment_ids: ["s1", "s2"],
    before: [{ internal_id: "s1" }, { internal_id: "s2" }],
    after: [{ internal_id: "s1" }, { internal_id: "s2" }],
  });
  assert.deepEqual(pending.affectedSegmentIds, ["s1", "s2"]);
  assert.equal(pending.before.length, 2);
  assert.equal(pending.after.length, 2);
});

test("adapter accepts only an internal handle", () => {
  const adapter = new SemanticCalibrationAdapter({ postPending: () => {} });
  adapter.model = buildTimelineModel(semantic);
  assert.throws(() => adapter.beginBoundaryDrag(0, "s1", 20), /internal boundary/);
  assert.deepEqual(adapter.beginBoundaryDrag(1, "s2", 60), {
    boundary_index: 1,
    actor_segment_id: "s2",
    new_frame_exclusive: 60,
  });
});
