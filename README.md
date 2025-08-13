# Project Title

## 🔄 Current Status

- **Main Active Branch**: [`main`](https://github.com/Westview-USC-Biomechanics-Collaboration/Prepare2Play-Learning-Experience/tree/chase-GUI)  
  _This is the most up-to-date working branch with the latest features and updates._

---

## File Outline
```
Prepare2Play/
│
├── main.py # App entry point — launches DisplayApp
│
├── gui/
│ ├── app.py # DisplayApp class: central GUI controller → initializes all the UI
│ ├── layout/ # UI builders
│ │ ├── canvas_manager.py # Everything canvas related (e.g. inits) + keyPress logic
│ │ ├── button_panel.py
│ │ ├── timeline_manager.py
│ │ └── background.py
│ ├── models/ # OOP data containers
│ │ ├── video_state.py
│ │ ├── force_state.py
│ │ └── state_manager.py # Contains flags + app state
│ ├── callbacks/ # Event handlers (buttons)
│ │ └── (all your callback files here)
│
├── processing/ # Force/LED sync processing + Rectangle detection (for 8 points)
│ ├── led_sync.py # To integrate after testing
│ ├── rect_detect.py # To create and integrate
│
├── vector_overlay/ # No need to change here
│ └── (unchanged)
│
├── utils/ # File formatting, conversion helpers
│ ├── file_io.py
│ └── frame_converter.py
```

## ✅ Outstanding Tasks

- [ ] Implement correct vector overly for side and top view
- [ ] Auto detections for force plate corners (assuming they are always in the same location)
- [ ] Implement direct processing from LED syncing to GUI
- [ ] Modularize Vector Overlay Code

---

## 🧪 Actively Developed Branches

| Branch | Description |
|--------|-------------|
| [`GUI for LED Syncing`](https://github.com/your-org/your-repo/tree/dev](https://github.com/Westview-USC-Biomechanics-Collaboration/Prepare2Play-Learning-Experience/tree/testingLED)) | (Deren) Takes in raw .csv file, syncs using LED, outputs vector overlay. Currently only working for top view   |

---

## 🛠️ Installation

1. Download Python 3.11.X from: https://www.python.org/downloads/  
2. Download Git from: https://git-scm.com/download/  
3. Clone the repository and open the folder in your preferred IDE or terminal.
4. Run the program and select **"create .venv file"**, choosing the correct Python version installed on your system.
5. Open your terminal and run:
   ```bash
   pip install -r requirements.txt


## 👥 Contributors

- Aarav Yadav  
- Chase Chen  
- Ayaan Irshad  
- Jessie Bao  
- Deren Erdem  
- James Guo  
- Rayyan Hussain  
- Nishk Shah  
- Breanna Thayillam  
- Christopher Yuan
