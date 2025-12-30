# Project Title

## 🛠️ Installation

1. Download Python 3.10.11 from: https://www.python.org/downloads/  
2. Download Git from: https://git-scm.com/download/  
3. Clone the repository and open the folder in your preferred IDE or terminal.
4. Run the program and select **"create .venv file"**, choosing the correct Python version installed on your system.
5. Open your terminal and run:
   ```bash
   pip install -r requirements.txt
6. Open terminal and run: winget install --id Gyan.FFmpeg -e

# Changes
## Nishk - Date: 12/30/2025
- Graphs colors are "opposite" so "red" as the input will display blue in the graph
- Saved PNG of the first frame
- SIDE VIEW:
  - Corrected side view crops for LED detection
- TOP VIEW:
  - Corrected res (1920x1080)
  - Changed LED crop
  - Changed graph colors
  - Added resultant horizontal force and vertical force (Fz)
    - Purple and Orange for FP1 and FP2, respectively
- TO-DO:
  - Manual input for the four corners
  - Manual input for the LED location
  - Vector colors for TOP VIEW should not be hard-coded (go in vectoroverlay_GUI.py, draw vectors method)
  - Side view auto detection of LED for the force plates 
  - Change naming convention from SHORT VIEW to SIDE VIEW in all the code
    - Add SIDE VIEW to config + fix is_side1 boolean input for the parameter
