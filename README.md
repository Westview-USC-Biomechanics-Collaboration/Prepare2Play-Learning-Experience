# Prepare To Play
# Table of Contents
* [Installation](#installation)
  * [Windows](#windows)
  * [MacOs](#macos)
# Installation
## Windows
> [!NOTE]  
> Most of the steps are for a first time install.
> If you have already completed this process and are trying to restart the program, skip to the final step.
### Install UV for Windows.
* Press WIN+R to open the RUN dialog.
* In the RUN dialog enter the following command.
```ps1
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
### Install Python 3.12.
* Press WIN+R to open the RUN dialog.
* In the RUN dialog enter the following command.
```ps1
uv python install 3.12
```
### Install Git for Windows. 
* Press WIN+R to open the RUN dialog.
* In the popup paste the following command.
```ps1
winget install -e --id Git.Git
```
### Clone the repository.
* Open file explorer and navigate to the folder you would like to use.
* Right click and click open in powershell.
* In the powershell window enter the following commands separately.
```ps1
git clone "https://github.com/Westview-USC-Biomechanics-Collaboration/Prepare2Play-Learning-Experience.git"

cd Prepare2Play-Learning-Experience
```
### Install Program Packages
* Close powershell and reopen it in the same way to restart powershell (needed after installing something).
* Enter the following command
```ps1
uv sync
```
### Running the Program
* Open powershell in the directory the program is installed in and enter the following command:
```ps1
uv run GUI/simpleGUI.py
```
### MacOS
#### Open Terminal
* Click the Launchpad icon, in the Dock, type Terminal in the search field, then click Terminal.
#### Install UV
* In the terminal enter the following command.
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
#### Restart the Terminal
* Close and reopen the terminal.

### Install Python 3.12.
* In the terminal enter the following command.
```bash
uv python install 3.12
```
### Install Git for MacOS. 
* In the terminal enter the following command.
```bash
brew install git
```
### Clone the repository.
* Open finder and navigate to the folder you would like to use.
* Click Finder in the top menu bar. Select Services > Services Settings > Files and Folders > Check "New Terminal at Folder" 
* Enter the following commands separately.
```bash
git clone "https://github.com/Westview-USC-Biomechanics-Collaboration/Prepare2Play-Learning-Experience.git"

cd Prepare2Play-Learning-Experience
```
### Install Program Packages
* Enter the following command
```bash
uv sync
```
### Running the Program
* Open terminal in the directory the program is installed in and enter the following command:
```bash
uv run GUI/simpleGUI.py
```

<!--
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
- Deleted old buttons (label force and label video) and added new ones for user selection of Male or Female COM -->