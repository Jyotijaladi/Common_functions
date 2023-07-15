import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
import sys
from C import *
# Function to execute the selected program
def execute_program(program, *args):
    try:
        # Redirect stdout to the ScrolledText widget
        output_text.delete("1.0", tk.END)  # Clear the output
        sys.stdout = StdoutRedirector(output_text)
        
        # Prepare the arguments for the exec statement
        args_str = ", ".join(map(repr, args))
        
        
        # Execute the program with the provided arguments
        exec(f"print({program}({2}))")
    except Exception as e:
        # Print the error message to the ScrolledText widget
        output_text.delete("1.0", tk.END)  # Clear the output
        output_text.insert(tk.END, f"Error executing program: {e}")
    finally:
        # Restore stdout to its default value
        sys.stdout = sys.__stdout__


# Custom stdout redirector
class StdoutRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, message):
        self.text_widget.insert(tk.END, message)
        self.text_widget.see(tk.END)

# Function to handle subject button click
def subject_click(subject):
    program_list.delete(0, tk.END)  # Clear the program list
    program_file = f"{subject}_programs.txt"  # File name based on subject
    try:
        with open(program_file, "r") as file:
            programs = file.read().splitlines()
        for program in programs:
            program_list.insert(tk.END, program)
    except FileNotFoundError:
        output_text.delete("1.0", tk.END)  # Clear the output
        output_text.insert(tk.END, f"File {program_file} not found.")

# Create the main window
window = tk.Tk()
window.title("Program Executor")

# Create a frame to hold the elements
frame = tk.Frame(window)
frame.pack(padx=10, pady=10, side=tk.LEFT)

# Create subject buttons
subjects = ["AI_ml", "Data mining", "Python", "C", "Java"]
for i, subject in enumerate(subjects):
    button = tk.Button(frame, text=subject, command=lambda s=subject: subject_click(s))
    button.grid(row=0, column=i, padx=10, pady=10, sticky="w")
execute_button = ttk.Button(frame, text="Execute", command=lambda: execute_program(program_list.get(tk.ACTIVE),Input_text.get()))

execute_button.grid(row=3,column=1, padx=10, pady=10, sticky="w")

# Create a scrolled frame for program list
scroll_frame = ttk.Frame(window)
scroll_frame.pack(pady=10, side=tk.LEFT)

# Create scrollbars
xscrollbar = ttk.Scrollbar(scroll_frame, orient=tk.HORIZONTAL)
xscrollbar.pack(side=tk.BOTTOM, fill=tk.X)
yscrollbar = ttk.Scrollbar(scroll_frame)
yscrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# Create program listbox
program_list = tk.Listbox(scroll_frame, xscrollcommand=xscrollbar.set, yscrollcommand=yscrollbar.set)
program_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Configure scrollbars
xscrollbar.config(command=program_list.xview)
yscrollbar.config(command=program_list.yview)

# Create a frame for the output
output_frame = ttk.Frame(window)
output_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True, side=tk.RIGHT)
# Create a label for the output
Input_label = ttk.Label(output_frame, text="Input:")
Input_label.pack()

# Create a ScrolledText widget to display the output
Input_text = tk.Entry(output_frame)
Input_text.pack(fill=tk.BOTH, expand=True)

# Create a label for the output
output_label = ttk.Label(output_frame, text="Output:",)
output_label.pack()

# Create a ScrolledText widget to display the output
output_text = ScrolledText(output_frame, wrap=tk.WORD,height=5,width=20)
output_text.pack(fill=tk.BOTH, expand=True)


# Start the GUI event loop
window.mainloop()
