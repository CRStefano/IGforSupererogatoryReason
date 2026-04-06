# IGforSupererogatoryReason
An application of the IG criterion taken from Shannon's information theory to capture in theory which reasons (p) are decision-making and which are supererogatory for a conclusion to be explained (q).

# ATTENTION, THERE IS ONE IMPORTANT PREREQUISITE:
“Knowing how to read”

# FILES LIST

You have everything you need here:

- IG_Analyzer_Setup.dmg: The executable for macOS.
- IG_Analyzer.exe: The executable for Windows.
- IGFOREASON_NICEGUIvim.py: The Python source code. This new version uses NiceGUI—make sure to install all the necessary dependencies!
- breast-cancer.csv: A sample CSV file ready for use.

# INSTRUCTIONS

How to use it?
Simply run the executable. A local web interface will open in your browser (the developer clearly has great taste!).

1. Upload: In the top right corner, upload your CSV file.
2. Configuration: Once uploaded, the system will ask for your search parameters:
      2a. Select what you want to explain (q).
      2b. Select the data type for the q column.
      2c. Set the instance and the value threshold (>=).
      2d. Filter Features: Below, you can uncheck (de-flag) the columns you don't want to use as explanations (p).   
4. Run: Click OK to confirm the settings. You can double-check the loaded data in the "DATA" tab.
5. Execution: To calculate the supererogatory reasons using Shannon IG, simply click "Start/Execution".
6. Et voilà! You will get your decision-making and supererogatory reasons. You can find them both in the interactive chart and in the left-hand container, complete with all their IG/Entropy values.

Bonus Features:
- Real-time Log: A log at the bottom tracks every single step of the process.
- Interactive Charts: You can zoom into the graph and even download it as an image.

Have fun!

#LINK FOR DOWNLOAD: https://github.com/CRStefano/IGforSupererogatoryReason/releases/tag/IGFOREASON
