import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

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

test("timeline positions 120-frame intervals proportionally without stacking", () => {
  const model = buildTimelineModel({
    timeline: {
      frame_count: 600,
      fps: 30,
      segments: [
        { internal_id: "s1", start_frame: 0, end_frame_exclusive: 120, text_cn: "准备" },
        { internal_id: "s2", start_frame: 120, end_frame_exclusive: 168, text_cn: "曝光异常" },
        { internal_id: "s3", start_frame: 168, end_frame_exclusive: 600, text_cn: "继续" },
      ],
    },
  });
  assert.equal(model.segments[1].leftPercent, 20);
  assert.equal(model.segments[1].widthPercent, 8);
  assert.equal(model.handles[0].leftPercent, 20);
  assert.equal(model.handles[1].leftPercent, 28);

  const markup = renderTimelineMarkup(model);
  assert.match(markup, /--segment-left:20%;--segment-width:8%/);
  assert.match(markup, /--boundary-left:20%/);
  assert.match(markup, /--boundary-left:28(?:\.0+)?%/);

  const css = readFileSync(new URL("./workbench.css", import.meta.url), "utf8");
  assert.match(css, /\.timeline-segment[^}]*left:\s*var\(--segment-left\)/s);
  assert.match(css, /\.timeline-segment[^}]*width:\s*var\(--segment-width\)/s);
  assert.match(css, /\.boundary-handle[^}]*left:\s*var\(--boundary-left\)/s);
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
  assert.equal(preview.following.leftPercent, Number(((60 / 195) * 100).toFixed(6)));
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

test("boundary and text pending rows expose the actual before and after values", () => {
  const slot = { innerHTML: "", querySelector() { return null; } };
  const adapter = new SemanticCalibrationAdapter();
  adapter.root = {
    querySelector(selector) { return selector === ".semantic-pending-slot" ? slot : null; },
  };

  adapter.renderPending({
    edit_type: "boundary",
    affected_segment_ids: ["s1", "s2"],
    before: [
      { internal_id: "s1", start_frame: 0, end_frame_exclusive: 120 },
      { internal_id: "s2", start_frame: 120, end_frame_exclusive: 168 },
    ],
    after: [
      { internal_id: "s1", start_frame: 0, end_frame_exclusive: 130 },
      { internal_id: "s2", start_frame: 130, end_frame_exclusive: 168 },
    ],
  });
  assert.match(slot.innerHTML, /修改前：0–119/);
  assert.match(slot.innerHTML, /修改后：0–129/);
  assert.match(slot.innerHTML, /修改前：120–167/);
  assert.match(slot.innerHTML, /修改后：130–167/);

  adapter.renderPending({
    edit_type: "text",
    affected_segment_ids: ["s2"],
    before: [{ internal_id: "s2", start_frame: 120, end_frame_exclusive: 168, text_cn: "拿杯子", text_en: "pick cup" }],
    after: [{ internal_id: "s2", start_frame: 120, end_frame_exclusive: 168, text_cn: "拿起杯子", text_en: "pick up cup" }],
  });
  assert.match(slot.innerHTML, /修改前中文：拿杯子/);
  assert.match(slot.innerHTML, /修改后中文：拿起杯子/);
  assert.match(slot.innerHTML, /修改前英文：pick cup/);
  assert.match(slot.innerHTML, /修改后英文：pick up cup/);
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

test("read-only tasks disable every boundary text and completion mutation control", () => {
  const textSlot = { innerHTML: "", querySelectorAll() { return []; } };
  const root = {
    innerHTML: "",
    querySelectorAll() { return []; },
    querySelector(selector) { return selector === ".semantic-text-slot" ? textSlot : null; },
  };
  const adapter = new SemanticCalibrationAdapter();
  adapter.render({ editable: false, semantic }, root);
  assert.match(root.innerHTML, /class="boundary-handle[^>]* disabled/);
  assert.match(textSlot.innerHTML, /data-action="save-text" disabled/);
  assert.match(root.innerHTML, /data-action="complete-semantic"[^>]* disabled/);
});
