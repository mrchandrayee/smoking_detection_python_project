import json, time
from pathlib import Path
import cv2
from .detectors import PersonDetector, SmokingDetector
from .alert_tracker import SmokingAlertTracker
from .visualizer import TechVisualizer, COLOR_PALETTE

class SmokingDetectionPipeline:
    def __init__(self,args):
        self.args=args
        cfg=json.loads(Path(args.config).read_text(encoding="utf-8"))
        self.cfg=cfg
        device=args.device
        self.person=PersonDetector(args.person_weights,cfg["person_conf"],cfg["person_iou"],device)
        resolved_device=self.person.device
        self.smoking=SmokingDetector(args.smoking_weights,cfg["smoking_conf"],cfg["smoking_iou"],resolved_device)
        self.alerts=SmokingAlertTracker(cfg["window_size"],cfg["alert_cooldown_sec"],cfg["alert_threshold"])
        self.visual=TechVisualizer(cfg["ui_scale"])
        self.snapshot_dir=Path(cfg["snapshot_dir"])
        if args.save_snapshots: self.snapshot_dir.mkdir(parents=True,exist_ok=True)
        self.event_log=[]

    def run(self):
        src=int(self.args.source) if str(self.args.source).isdigit() else self.args.source
        cap=cv2.VideoCapture(src)
        if not cap.isOpened(): raise RuntimeError(f"Cannot open source: {self.args.source}")
        w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        fps_in=cap.get(cv2.CAP_PROP_FPS)
        fps_in=fps_in if fps_in and fps_in>1 else 25.0
        total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out=Path(self.args.output); out.parent.mkdir(parents=True,exist_ok=True)
        writer=cv2.VideoWriter(str(out),cv2.VideoWriter_fourcc(*"mp4v"),fps_in,(w,h))
        if not writer.isOpened(): raise RuntimeError(f"Cannot create output: {out}")
        frame_idx=0; start=time.time(); fps_smooth=0.0
        print("[INFO] Pipeline started. Press q in preview window to stop.")
        try:
            while True:
                ok,frame=cap.read()
                if not ok: break
                frame_idx+=1; t0=time.time()
                persons=self.person.detect(frame)
                smoking_count=0
                for person in persons:
                    x1,y1,x2,y2=person["box"]
                    pad=int((y2-y1)*self.cfg["crop_padding"])
                    xa=max(0,x1-pad); ya=max(0,y1-pad)
                    xb=min(w,x2+pad); yb=min(h,y2+pad)
                    crop=frame[ya:yb,xa:xb]
                    detections=self.smoking.detect_in_crop(crop)
                    person_smoking=bool(detections)
                    smoking_count+=len(detections)
                    self.visual.draw_tech_box(frame,person["box"],"PERSON",
                        person["conf"],COLOR_PALETTE["person"],style="person")
                    for det in detections:
                        dx1,dy1,dx2,dy2=det["box"]
                        global_box=(xa+dx1,ya+dy1,xa+dx2,ya+dy2)
                        label=det["label"].upper()
                        key="cigarette" if "cig" in label else "smoking"
                        self.visual.draw_tech_box(frame,global_box,label,det["conf"],
                            COLOR_PALETTE.get(key,COLOR_PALETTE["smoking"]),style="rect")
                state=self.alerts.update(smoking_count)
                if state["new_alert"]:
                    event={"frame":frame_idx,"time_sec":round(frame_idx/fps_in,3),
                           "status":state["status"],"smoking_count":smoking_count,
                           "occurrence":state["smoking_occurrences"]}
                    self.event_log.append(event)
                    if self.args.save_snapshots:
                        snap=self.snapshot_dir/f"smoking_alert_{frame_idx:07d}.jpg"
                        cv2.imwrite(str(snap),frame)
                    print(f'[ALERT] Smoking detected at {event["time_sec"]:.2f}s '
                          f'(occurrence {event["occurrence"]})')
                elapsed=max(time.time()-t0,1e-6)
                current_fps=1.0/elapsed
                fps_smooth=current_fps if fps_smooth==0 else .9*fps_smooth+.1*current_fps
                frame=self.visual.draw_dashboard(frame,state["status"],len(persons),
                    smoking_count,fps_smooth,frame_idx,total,state["smoking_occurrences"])
                writer.write(frame)
                if self.args.display:
                    cv2.imshow("Smoking Detection",frame)
                    if cv2.waitKey(1)&0xFF==ord("q"): break
        finally:
            cap.release(); writer.release()
            if self.args.display: cv2.destroyAllWindows()
        report={"source":str(self.args.source),"frames_processed":frame_idx,
                "duration_sec":round(frame_idx/fps_in,3),
                "elapsed_sec":round(time.time()-start,2),
                "smoking_model_enabled":self.smoking.enabled,
                "smoking_occurrences":self.alerts.smoking_occurrences,
                "events":self.event_log}
        report_path=out.with_name(out.stem+"_events.json")
        report_path.write_text(json.dumps(report,indent=2),encoding="utf-8")
        print(f"[DONE] Annotated video: {out}")
        print(f"[DONE] Event report: {report_path}")
