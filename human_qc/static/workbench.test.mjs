import test from "node:test";
import assert from "node:assert/strict";

import {
  SemanticCalibrationAdapter,
  buildTimelineModel,
  makeBoundaryPayload,
  pendingPresentation,
  renderTimelineMarkup,
} from "./semantic_adapter.js";
import { WorkbenchApp, mutationControlsDisabled } from "./app.js";

const segments = [
  { internal_id: "s1", start_frame: 0, end_frame_exclusive: 51, text_cn: "接近杯子", text_en: "approach" },
  { internal_id: "s2", start_frame: 51, end_frame_exclusive: 123, text_cn: "拿起杯子", text_en: "pick up" },
  { internal_id: "s3", start_frame: 123, end_frame_exclusive: 195, text_cn: "放入托盘", text_en: "place" },
];

const task = {
  asset_id: "asset-1",
  revision: 4,
  semantic: {
    report_revision: 4,
    report_state: "in_progress",
    timeline: { frame_count: 195, fps: 30, segments },
    pending_edit: null,
  },
};

test("N segments render exactly N-1 internal shared-boundary handles", () => {
  const model = buildTimelineModel(task.semantic);
  assert.equal(model.segments.length, 3);
  assert.equal(model.handles.length, 2);
  assert.deepEqual(model.handles.map((item) => item.boundary_index), [1, 2]);

  const markup = renderTimelineMarkup(model);
  assert.equal((markup.match(/class="boundary-handle/g) || []).length, 2);
  assert.equal((markup.match(/class="timeline-segment/g) || []).length, 3);
  assert.doesNotMatch(markup, /class="timeline-segment[^>]*draggable=/);
  assert.match(markup, /data-end-frame="50"/);
  assert.match(markup, /data-end-frame="122"/);
  assert.doesNotMatch(markup, /data-end-frame="123"/);
});

test("boundary payload carries shared boundary index, actor segment, and exclusive frame", () => {
  assert.deepEqual(
    makeBoundaryPayload({
      boundaryIndex: 1,
      actorSegmentId: "s2",
      frameExclusive: 60,
      expectedRevision: 4,
      leaseToken: "lease-1",
    }),
    {
      boundary_index: 1,
      actor_segment_id: "s2",
      new_frame_exclusive: 60,
      expected_revision: 4,
      lease_token: "lease-1",
    },
  );
});

test("pending presentation includes both affected before/after snapshots and locks edits", () => {
  const pendingTask = {
    ...task,
    semantic: {
      ...task.semantic,
      pending_edit: {
        edit_type: "boundary",
        affected_segment_ids: ["s1", "s2"],
        before: [
          { internal_id: "s1", start_frame: 0, end_frame_exclusive: 51 },
          { internal_id: "s2", start_frame: 51, end_frame_exclusive: 123 },
        ],
        after: [
          { internal_id: "s1", start_frame: 0, end_frame_exclusive: 60 },
          { internal_id: "s2", start_frame: 60, end_frame_exclusive: 123 },
        ],
      },
    },
  };
  const view = pendingPresentation(pendingTask.semantic.pending_edit);
  assert.deepEqual(view.affectedSegmentIds, ["s1", "s2"]);
  assert.deepEqual(view.before.map((item) => item.end_frame_exclusive), [51, 123]);
  assert.deepEqual(view.after.map((item) => item.start_frame), [0, 60]);
  assert.equal(mutationControlsDisabled(pendingTask), true);
});

test("adapter accepts boundary drag payloads only for internal handles", () => {
  const adapter = new SemanticCalibrationAdapter({
    postPending: () => {},
  });
  adapter.model = buildTimelineModel(task.semantic);
  assert.throws(() => adapter.beginBoundaryDrag(0, "s1", 20), /internal boundary/);
  assert.throws(() => adapter.beginBoundaryDrag(3, "s3", 170), /internal boundary/);
  assert.deepEqual(adapter.beginBoundaryDrag(1, "s2", 60), {
    boundary_index: 1,
    actor_segment_id: "s2",
    new_frame_exclusive: 60,
  });
});

test("WorkbenchApp keeps server-conflict errors visible without overwriting the task", async () => {
  const app = new WorkbenchApp({
    fetcher: async () => ({
      ok: false,
      status: 409,
      json: async () => ({ error: { code: "stale_revision", message: "refresh" } }),
    }),
  });
  app.applyServerTask(task);
  await assert.rejects(() => app.requestTask("asset-1"), /refresh/);
  assert.equal(app.task.revision, 4);
  assert.equal(app.lastError.code, "stale_revision");
});
