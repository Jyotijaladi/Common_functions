import code
import importlib
import inspect
import hashlib
import queue
import sys
import threading
import tkinter as tk
import traceback
from tkinter.scrolledtext import ScrolledText
from concurrent.futures import ThreadPoolExecutor, as_completed

class Pipe:
    """mock stdin stdout or stderr"""

    def __init__(self):
        self.buffer = queue.Queue()
        self.reading = False

    def write(self, data):
        self.buffer.put(data)

    def flush(self):
        pass

    def readline(self):
        self.reading = True
        line = self.buffer.get()
        self.reading = False
        return line


class Console(tk.Frame):
    """A tkinter widget which behaves like an interpreter"""

    def __init__(self, parent, _locals, exit_callback):
        super().__init__(parent)

        self.module_var = tk.StringVar()
        self.configure(background="white")
        self.text_widget = ScrolledText(self, height=10)
        self.text_widget.pack()
        self.text_widget.configure(border=True, borderwidth=2, selectborderwidth=2)
        self.frame_b = tk.Frame(self)
        self.frame_b.pack()
        self.frame_b.configure(background="white")
        button_row = 1
        grid = {1: (1, 1), 2: (1, 2), 3: (2, 1), 4: (2, 2), 5: (3, 1)}
        self.module_buttons = {}
        for module_name in ["AI_ML", "Java_Programs", "Python_lib", "Data_Mining", "C"]:
            button = tk.Button(self.frame_b, width=11, text=module_name, command=lambda name=module_name: self.module_button_clicked(name))
            button.grid(row=grid[button_row][0], column=grid[button_row][1], padx=5, pady=5)
            self.module_buttons[module_name] = button
            button_row += 1
        call_button = tk.Button(self.frame_b, width=11, text="Call Function", command=self.call_selected_function)
        call_button.grid(row=3, column=2, padx=5, pady=5)
        clear_button = tk.Button(self.frame_b, width=11, text="Clear Screen", command=self.clear_screen)
        clear_button.grid(row=4, column=2, padx=5, pady=5)
        self.text = ConsoleText(self, wrap=tk.WORD)
        self.text.pack(fill=tk.BOTH, expand=True)

        self.shell = code.InteractiveConsole(_locals)

        # make the enter key call the self.enter function
        self.text.bind("<Return>", self.enter)
        self.prompt_flag = True
        self.command_running = False
        self.exit_callback = exit_callback

        # replace all input and output
        sys.stdout = Pipe()
        sys.stderr = Pipe()
        sys.stdin = Pipe()

        def loop():
            self.read_from_pipe(sys.stdout, "stdout")
            self.read_from_pipe(sys.stderr, "stderr", foreground='red')

            self.after(50, loop)

        self.after(50, loop)

        # Store imported modules
        self.modules = {}
        self.thread_pool = ThreadPoolExecutor(max_workers=5)  # You can adjust the number of worker threads here
    def clear_screen(self):
        self.text_widget.delete(1.0, tk.END)
        self.text.delete(1.0, tk.END)
        self.text.update_idletasks()
        self.text_widget.update_idletasks()  # Update the display immediately

        # Clear the prompt flag to start a new prompt after clearing
        self.enter(e=sys)

        

    def import_module_async(self, module_name):
        if module_name not in self.modules:
            self.thread_pool.submit(self.import_module, module_name)

    def import_module(self, module_name):
        try:
            module = importlib.import_module(module_name)
            self.modules[module_name] = module
        except ImportError:
            traceback.print_exc()

    def call_selected_function(self):
        selected_function = self.text_widget.get(tk.SEL_FIRST, tk.SEL_LAST).strip()
        selected_module = self.module_var.get()

        if selected_module not in self.modules:
            # Import the selected module in the background if it's not already imported
            self.import_module_async(selected_module)

        module = self.modules.get(selected_module)

        if module is not None:
            cmd = f"import {selected_module} as m; m.{selected_function}()"
            self.execute_code_in_thread(cmd)

    def execute_code_in_thread(self, code):
        self.text.write(code + "\n", "command")
        future = self.thread_pool.submit(self.run_code, code)
        future.add_done_callback(self.on_command_completed)

    def on_command_completed(self, future):
        try:
            future.result()
        except Exception as e:
            traceback.print_exc()

    def module_button_clicked(self, module_name):
        self.module_var.set(module_name)
        self.display_function_names(module_name)

    def display_function_names(self, module_name):
        self.text_widget.delete("1.0", tk.END)

        if module_name not in self.modules:
            # Import the selected module in the background if it's not already imported
            self.import_module_async(module_name)

        module = self.modules.get(module_name)

        if module is not None:
            function_names = [name for name, obj in inspect.getmembers(module, inspect.isfunction)
                              if inspect.getmodule(obj) == module]

            for name in function_names:
                self.text_widget.insert(tk.END, name + "\n")

    def run_code(self, code):
        self.text.insert(tk.END, code + "\n")
        self.enter(e=sys)

    def prompt(self):
        self.prompt_flag = True

    def read_from_pipe(self, pipe: Pipe, tag_name, **kwargs):
        if self.prompt_flag and not self.command_running:
            self.text.prompt()
            self.prompt_flag = False

        string_parts = []
        while not pipe.buffer.empty():
            part = pipe.buffer.get()
            string_parts.append(part)

        str_data = ''.join(string_parts)
        if str_data:
            if self.command_running:
                insert_position = "end-1c"
            else:
                insert_position = "prompt_end"

            self.text.write(str_data, tag_name, insert_position, **kwargs)

    def enter(self, e):
        if sys.stdin.reading:
            line = self.text.consume_last_line()
            line = line + '\n'
            sys.stdin.buffer.put(line)
            return

        if self.command_running:
            return

        command = self.text.read_last_line()
        try:
            compiled = code.compile_command(command)
            is_complete_command = compiled is not None
        except (SyntaxError, OverflowError, ValueError):
            self.text.consume_last_line()
            self.prompt()
            traceback.print_exc()
            return

        if is_complete_command:
            self.text.consume_last_line()

            self.prompt()
            self.command_running = True

            def run_command():
                try:
                    self.shell.runcode(compiled)
                except SystemExit:
                    self.after(0, self.exit_callback)

                self.command_running = False

            threading.Thread(target=run_command).start()


class ConsoleText(ScrolledText):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        def on_modified(event):
            flag = self.edit_modified()
            if flag:
                self.after(10, self.on_text_change(event))
            self.edit_modified(False)

        self.bind("<<Modified>>", on_modified)

        self.console_tags = []
        self.mark_set("prompt_end", 1.0)
        self.committed_hash = None
        self.committed_text_backup = ""
        self.commit_all()

    def prompt(self):
        self.mark_set("prompt_end", 'end-1c')
        self.mark_gravity("prompt_end", tk.LEFT)
        self.write(">>> ", "prompt", foreground="blue")
        self.mark_gravity("prompt_end", tk.RIGHT)

    def commit_all(self):
        self.commit_to('end-1c')

    def commit_to(self, pos):
        if self.index(pos) in (self.index("end-1c"), self.index("end")):
            self.mark_set("committed_text", "end-1c")
            self.mark_gravity("committed_text", tk.LEFT)
        else:
            for i, (tag_name, _, _) in reversed(list(enumerate(self.console_tags))):
                if tag_name == "prompt":
                    tag_ranges = self.tag_ranges("prompt")
                    self.console_tags[i] = ("prompt", tag_ranges[-2], tag_ranges[-1])
                    break

        self.committed_hash = self.get_committed_text_hash()
        self.committed_text_backup = self.get_committed_text()

    def get_committed_text_hash(self):
        return hashlib.md5(self.get_committed_text().encode()).digest()

    def get_committed_text(self):
        return self.get(1.0, "committed_text")

    def write(self, string, tag_name, pos='end-1c', **kwargs):
        start = self.index(pos)
        self.insert(pos, string)
        self.see(tk.END)
        self.commit_to(pos)
        self.tag_add(tag_name, start, pos)
        self.tag_config(tag_name, **kwargs)
        self.console_tags.append((tag_name, start, self.index(pos)))

    def on_text_change(self, event):
        if self.get_committed_text_hash() != self.committed_hash:
            self.mark_gravity("committed_text", tk.RIGHT)
            self.replace(1.0, "committed_text", self.committed_text_backup)
            self.mark_gravity("committed_text", tk.LEFT)

            for tag_name, start, end in self.console_tags:
                self.tag_add(tag_name, start, end)

    def read_last_line(self):
        return self.get("committed_text", "end-1c")

    def consume_last_line(self):
        line = self.read_last_line()
        self.commit_all()
        return line


if __name__ == '__main__':
    root = tk.Tk()
    root.config(background="red")
    main_window = Console(root, locals(), root.destroy)
    main_window.pack(fill=tk.BOTH, expand=True)
    root.mainloop()
