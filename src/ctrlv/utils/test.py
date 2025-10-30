# import json, cv2
# import os
# clip="/no_backups/s1492/Ctrl-V/kitti_preprocessed_ctrlv/train/clip_00000"
# ann=json.load(open(f"{clip}/annotations.json"))
# f=ann["frames"][0]
# img=cv2.imread(os.path.join(clip, f["image_path"]))
# for o in f["objects"]:
#     x1,y1,x2,y2=map(int,o["bbox_2d"])
#     cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
# cv2.imwrite(f"{clip}/_debug_overlay_2.png",img)
import json, cv2, os

# === USER SETTINGS ===
clip = "/no_backups/s1492/Ctrl-V/kitti_preprocessed_ctrlv/train/clip_00001"
frame_idx = 0             # choose which frame index from annotations.json
include_types = {"car"}   # choose which object types to visualize (e.g. {"car","truck"})
include_ids = None        # optionally filter by specific IDs, e.g. {1, 5, 7}
border_color = (0, 255, 0)
border_thickness = 2
save_path = f"{clip}/_debug_overlay_filtered.png"

# ======================
ann = json.load(open(f"{clip}/annotations.json"))
frames = ann["frames"]
if frame_idx >= len(frames):
    raise IndexError(f"Frame index {frame_idx} out of range ({len(frames)} total)")

f = frames[frame_idx]
img_path = os.path.join(clip, f["image_path"])
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Could not read image at {img_path}")

drawn = 0
for o in f["objects"]:
    o_type = o.get("type", "").lower()
    o_id = o.get("id", -1)

    # Apply filtering
    if include_types and o_type not in include_types:
        continue
    if include_ids and o_id not in include_ids:
        continue

    # Draw bbox
    x1, y1, x2, y2 = map(int, o["bbox_2d"])
    cv2.rectangle(img, (x1, y1), (x2, y2), border_color, border_thickness)

    # Optional: label with type or ID
    label = f"{o_type}:{o_id}" if o_id >= 0 else o_type
    cv2.putText(img, label, (x1 + 2, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, border_color, 1, cv2.LINE_AA)
    drawn += 1

cv2.imwrite(save_path, img)
print(f" Saved overlay with {drawn} objects → {save_path}")
