# Install process:
 - download repository.
 - update MikTex package manager.

# Opt1: Copy whole template in your project documentation folder
 1. open doc_settings.txt and adapt according to your needs (a maxiumum number of 5 authors is possible). 
 2. insert your figures in the respective figs folder. 
 3. insert your latex chapters in the respective chapters folder.
 4. Compile in Editor or run make.bat. 

# Opt2: Copy content of folder "doc/src" in your project documentation folder   
1. set MikTex path to depending files
 - open MikTex Console
 - switch to administrator mode
 - Go to settings -> Directories and add Path to ait_report repository
2. Follow the steps from Opt1 1-4. 


 # useful batch scripts
 - running clean.bat deletes all temporary files. 
 - running make.bat compiles the main.tex file. 

# Bug report
 - If you find any bugs or have suggestions for improvements, please contact Amadeus Lobe
