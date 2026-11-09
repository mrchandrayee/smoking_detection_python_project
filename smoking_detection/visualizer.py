import cv2
import numpy as np

COLOR_PALETTE = {
    "person": (255, 191, 0),
    "smoking": (0, 0, 255),
    "cigarette": (0, 220, 255),
    "smoke": (0, 220, 255),
    "default": (0, 255, 255),
}
STATUS_COLORS = {"SAFE": (50,180,50), "WARNING": (0,0,230)}

class TechVisualizer:
    def __init__(self, scale=1.0, style="person"):
        self.scale = float(scale)
        self.style = style

    def draw_tech_box(self, img, box, label, conf, color, thickness=2, style="person"):
        x1,y1,x2,y2 = map(int, box)
        h,w = img.shape[:2]
        x1,y1=max(0,x1),max(0,y1); x2,y2=min(w-1,x2),min(h-1,y2)
        if x2 <= x1 or y2 <= y1: return
        corner_len = max(8, int(min(x2-x1,y2-y1)*0.22))
        if style == "person":
            corners = [
                ((x1,y1),(x1+corner_len,y1)),((x1,y1),(x1,y1+corner_len)),
                ((x2,y1),(x2-corner_len,y1)),((x2,y1),(x2,y1+corner_len)),
                ((x1,y2),(x1+corner_len,y2)),((x1,y2),(x1,y2-corner_len)),
                ((x2,y2),(x2-corner_len,y2)),((x2,y2),(x2,y2-corner_len)),
            ]
            cv2.rectangle(img,(x1,y1),(x2,y2),(255,255,255),1)
            for p1,p2 in corners:
                cv2.line(img,p1,p2,(255,255,255),thickness+1,cv2.LINE_AA)
                cv2.line(img,p1,p2,color,thickness,cv2.LINE_AA)
        else:
            cv2.rectangle(img,(x1,y1),(x2,y2),(255,255,255),1)
            cv2.rectangle(img,(x1,y1),(x2,y2),color,thickness,cv2.LINE_AA)
        text = f"{label} {conf:.0%}"
        fs=max(.35,.55*self.scale); th=max(1,int(1.5*self.scale))
        (tw,thh),base=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,fs,th)
        ty=max(thh+6,y1-5)
        cv2.rectangle(img,(x1,ty-thh-5),(x1+tw+8,ty+base),(15,15,15),-1)
        cv2.putText(img,text,(x1+4,ty),cv2.FONT_HERSHEY_SIMPLEX,fs,color,th,cv2.LINE_AA)

    def draw_dashboard(self, img, status, person_count, smoking_count, fps,
                       frame_idx, total_frames, occurrences):
        h,w=img.shape[:2]
        scale=max(.55,min(1.4,h/720))
        banner_h=max(36,int(45*scale))
        status_color=STATUS_COLORS.get(status,(50,50,50))
        overlay=img.copy()
        cv2.rectangle(overlay,(0,0),(w,banner_h),status_color,-1)
        cv2.addWeighted(overlay,.78,img,.22,0,img)
        banner=("SYSTEM NORMAL - NO SMOKING DETECTED" if status=="SAFE" else
                f"ALERT: SMOKING DETECTED ({occurrences} OCCURRENCE(S))")
        fs=.55*scale; th=max(1,int(2*scale))
        cv2.putText(img,banner,(int(12*scale),int(banner_h*.68)),
                    cv2.FONT_HERSHEY_SIMPLEX,fs,(255,255,255),th,cv2.LINE_AA)
        card_x,card_y=int(12*scale),banner_h+int(12*scale)
        card_w,card_h=int(270*scale),int(125*scale)
        card_overlay=img.copy()
        cv2.rectangle(card_overlay,(card_x,card_y),(min(w-1,card_x+card_w),min(h-1,card_y+card_h)),(20,20,20),-1)
        cv2.addWeighted(card_overlay,.70,img,.30,0,img)
        lines=["SMOKING DETECTION V2",f"STATUS: {status}",
               f"PERSONS DETECTED: {person_count}",
               f"SMOKING / CIGARETTES: {smoking_count}",
               f"FPS: {fps:.1f} | FRAME: {frame_idx}/{total_frames if total_frames else '?'}"]
        y=card_y+int(22*scale)
        for line in lines:
            cv2.putText(img,line,(card_x+int(9*scale),y),cv2.FONT_HERSHEY_SIMPLEX,
                        .40*scale,(255,255,255),1,cv2.LINE_AA)
            y+=int(20*scale)
        return img
