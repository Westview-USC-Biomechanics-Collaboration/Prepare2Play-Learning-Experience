# Vector Overlay Processing Software V1

This is the first version of student created vector overlay software that has been in development from Summer 2022 to Summer 2026. This code works and has been used to process vector overlays for HPOE final projects : Spring 2025, Fall 2025, Spring 2026.  There are some problems with this version that have not been fixed.  1) Force data on the Force time graph is undersampled (only plotting 120fps).  2)body segment end points determined by Media Pipe are delayd, resulting in inaccurate COM positions for fast movements  3)

This code is difficult to follow, has been passed onto a number of software students without documentation, and is difficult to trubleshoot.  There is no clear flow.  Variable names are outdated or are used in multiple scenarios with no clear definitions.  The GUI is outdated and creates user input errors that can be eliminated.

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
  - Side view auto detection of LED for the force plates
  - Change naming convention from SHORT VIEW to SIDE VIEW in all the code
    - Add SIDE VIEW to config + fix is_side1 boolean input for the parameter
  - Vector colors for TOP VIEW should not be hard-coded (go in vectoroverlay_GUI.py, draw vectors method) (DONE)
  - Fix TOP VIEW graph axis (not written properly) (DONE)

## Nishk - Date 1/1/2025
- Vectors colors for TOP VIEW are NOT hardcoded anymore
- TOP VIEW graph axis are drawn properly 
- Deleted old buttons (label force and label video) and added new ones for user selection of Male or Female COM
