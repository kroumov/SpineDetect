from psychopy import visual, core, event
import math

# ==========================================
# 1. Setup Window and Stimuli
# ==========================================
# Create a window (set fullscr=True for actual experiments)
win = visual.Window(size=(800, 600), fullscr=False, color='gray', units='height')

# Create the central checkerboard 
# tex='sqr2D' creates the checkerboard pattern, sf controls how many squares
checkerboard = visual.GratingStim(win=win, tex='sqr2D', mask='none', size=0.5, sf=5)

# Create the distractor (Condition B's red ball traveling the periphery)
distractor = visual.Circle(win=win, radius=0.03, fillColor='red', lineColor='red')

# ==========================================
# 2. Experiment Parameters
# ==========================================
flicker_freq = 10.0  # The target SSVEP frequency in Hz
trial_duration = 10.0  # How long each condition lasts in seconds

def run_condition(condition_type):
    """
    Runs either Condition A (Control) or Condition B (Distracted)
    """
    clock = core.Clock()
    
    # Loop for the duration of the trial
    while clock.getTime() < trial_duration:
        t = clock.getTime()
        
        # --- SSVEP Flicker Mechanism ---
        # Contrast flips between 1 and -1 at the target frequency (Pattern Reversal)
        checkerboard.contrast = 1 if math.sin(2 * math.pi * flicker_freq * t) > 0 else -1
        checkerboard.draw()
        
        # --- Condition B: The Distractor ---
        if condition_type == 'B':
            # Calculate peripheral trajectory using sine and cosine
            # radius = 0.35 screen height units, speed = 0.3 Hz
            x = 0.35 * math.cos(2 * math.pi * 0.3 * t)
            y = 0.35 * math.sin(2 * math.pi * 0.3 * t)
            distractor.pos = (x, y)
            distractor.draw()
            
        # Flip the window to show the drawn stimuli
        win.flip()
        
        # Allow the user to escape the experiment early
        if event.getKeys(['escape', 'q']):
            win.close()
            core.quit()

# ==========================================
# 3. Run the Experiment Sequence
# ==========================================
# Condition A Setup
instruction = visual.TextStim(win, text="Condition A: Focus on the center.\nPress any key to start.", color='white')
instruction.draw()
win.flip()
event.waitKeys() # Wait for user to be ready

# Run Condition A (Checkerboard only)
run_condition('A') 

# Condition B Setup
instruction.text = "Condition B: Focus on the center while ignoring the red ball.\nPress any key to start."
instruction.draw()
win.flip()
event.waitKeys()

# Run Condition B (Checkerboard + Red Ball)
run_condition('B') 

# Clean up and close
win.close()
core.quit()