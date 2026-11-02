from multiprocessing import freeze_support

from birdnet_analyzer.gui import main

# A frozen spawn child runs this module too: dispatch it before main() imports the GUI.
freeze_support()
main()
