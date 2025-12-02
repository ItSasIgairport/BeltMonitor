# YOLO Project Configuration Guide

This project uses `config.json` to manage settings for network connections, video processing, alarms, and the user interface.

## Hot Reloading
Some parameters can be updated **while the application is running**. The system monitors `config.json` for changes and automatically applies updates for supported parameters (marked below).

---

## Configuration Parameters

### 1. Network Settings (`network`)
Controls camera connections and video source types.
* **`source_type`**: Defines the input source. Options:
  * `"stream"`: Connect to live IP cameras via RTSP/HTTP.
  * `"file"`: Use a local video file for testing.
* **`video_file_path`**: Path to the video file (used only if `source_type` is `"file"`).
* **`username`**: Username for camera authentication.
* **`password`**: Password for camera authentication.
* **`reconnect_delay`**: Seconds to wait before retrying a failed connection.
* **`camera_resolution`**: Resolution string sent to the camera URL (e.g., "1920x1080").
* **`cameras`**: A list of IP addresses for the cameras to connect to.
  * *Note*: Changing this requires a restart.

### 2. Processing Settings (`processing`)
Controls the YOLO model and image processing pipeline.
* **`fps`**: Target frame rate for the camera request.
* **`scale_percentage`** (🔥 **Hot Reloadable**):
  * Resize factor for processing frames (100 = original size).
  * Lower values improve performance but reduce detection accuracy for small objects.
* **`model_folder_path`**: Directory containing YOLO model weights.
* **`model_name`**: Filename of the YOLO model (e.g., `yolo11s-pose.pt`).
* **`model_confidence_threshold`** (🔥 **Hot Reloadable**):
  * Minimum confidence (0.0 - 1.0) required to consider a detection valid.
* **`keypoints`** (🔥 **Hot Reloadable**):
  * List of body keypoint indices to check against danger zones (polygons).
  * Common values: Shoulders (5,6), Hips (11,12), Knees (13,14), Ankles (15,16).

### 3. Alarm Settings (`alarm`)
Controls the logic for triggering security alerts.
* **`trigger_time`** (🔥 **Hot Reloadable**):
  * Duration (in seconds) a person must stay inside a "WARNING" zone before it escalates to an "ALARM".

### 4. Event Recording (`event_recording`)
Controls automatic video recording when events occur.
* **`enabled`** (🔥 **Hot Reloadable**): Master switch to enable/disable event-based recording.
* **`pre_event_sec`** (🔥 **Hot Reloadable**): Number of seconds of video to save *before* the event occurred (buffer).
* **`post_event_sec`** (🔥 **Hot Reloadable**): Number of seconds of video to save *after* the event trigger.
* **`trigger_on_warning`** (🔥 **Hot Reloadable**): If true, recording starts when status becomes "WARNING".
* **`trigger_on_alarm`** (🔥 **Hot Reloadable**): If true, recording starts when status becomes "ALARM".
* **`save_path`** (🔥 **Hot Reloadable**): Directory where event video clips will be saved.

### 5. User Interface (`ui`)
Controls the visualization window.
* **`enable_ui`**: Master switch to show/hide the video grid window.
* **`enable_polygon_ui`** (🔥 **Hot Reloadable**):
  * Toggles the on-screen "EDIT MODE" indicator and interaction logic for drawing polygons.
* **`queue_size`**: Size of the frame buffer for the UI thread (lower reduces latency).

### 6. Logging (`logging`)
* **`level`**: Logging verbosity. Options: `DEBUG`, `INFO`, `WARNING`, `ERROR`.

---

## How to Change Settings
1. Open `config.json` in a text editor.
2. Modify the desired value.
3. Save the file.
4. If the parameter is **Hot Reloadable**, the application will apply the change within ~2 seconds. Otherwise, restart the application.

