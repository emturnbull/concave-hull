"""

    Author:                Erin Turnbull
    Last Modified:        2019-03-27
    
    Provides a reusable printing utility class and functions (tweet.echo()) that can 
    print to multiple different locations at once.
    
"""

import datetime

class Printer:
    """ Manages printing operations"""

    DEBUG = -1 # Debug messages, crap that people don't want to see
    INFO = 0 # For the mildly curious
    WARNING = 1 # Your output might be in error
    ERROR = 2 # No really, you might want to reconsider what you did
    CRITICAL = 3 # You broke it

    class __FilePrinter:
        """ Responsible for printing to a file"""
        __slots__ = ['filename', 'handle']
        
        def __init__(self, file, append=True):
            """ Initializer""" 
            self.filename = file
            self.handle = open(self.filename, 'a+')
            
        def __del__(self):
            """ Deconstructor """
            close(self.handle)
            
        def echo(self, message):
            """ Prints a message to a file, with a new line. """
            self.handle.write(message + "\n")

    class __Printer:
        """ Singleton object responsible for printing """
        __slots__ = ['printers', 'log_files']
        
        def __init__(self):
            """ Constructor """
            self.printers = []
            self.log_files = []
            
        def __del__(self):
            """ Deconstructor """
            del self.log_files
            del self.printers
            
        def clear(self):
            """ Clears all the printers. """
            del self.printers
            self.printers = []
            del self.log_files
            self.log_files = []
            
        def register_printer_function(self, printer:callable, min_level:int=-10, max_level:int=10):
            """ Registers a function which will be called with echo() if the message level is in bounds """
            self.printers.append((printer, min_level, max_level))
            
        def register_log_file(self, log_file, append=True, min_level=-10, max_level=10):
            """ Registers a file via __FilePrinter(). """
            fp = Printer.__FilePrinter(log_file, append)
            self.log_files.append(fp)
            self.register_printer_function(fp.echo, min_level, max_level)
            
        def echo(self, message, level):
            """ Sends a message to all registered printers. """
            for printer, min_level, max_level in self.printers:
                if level >= min_level and level <= max_level:
                    printer(message)
    
    # Singleton instance variable
    instance = None
    
    def __init__(self):
        """ Constructor, singleton pattern. """
        if not Printer.instance:
            Printer.instance = Printer.__Printer()
    
    def level_name(self, level:int):
        """ Retrieves the English name of the message. To do: translations?"""
        if level == Printer.DEBUG:
            return "DBUG"
        elif level == Printer.INFO:
            return "INFO"
        elif level == Printer.WARNING:
            return "WARN"
        elif level == Printer.ERROR:
            return "ERRO"
        elif level == Printer.CRITICAL:
            return "CRIT"
        else:
            return "????"
    
    def format_message(self, message:str, level:int):
        """ Formats messages with date, time, etc. Very pretty. """
        return "{} [{}]: {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), self.level_name(level), message)
        
    def register_log_file(self, log_file:str, append:bool=True, min_level:int=-10, max_level:int=10):
        """ Registers a log file to participate in printing. """
        Printer.instance.register_log_file(log_file, append, min_level, max_level)
        
    def register_printer_function(self, printer, min_level=-10, max_level=10):
        """ Registers any function to participate in printing. """
        Printer.instance.register_printer_function(printer, min_level, max_level)
        
    def echo(self, message, level=0):
        """ Sends a message to all the printers. """
        Printer.instance.echo(self.format_message(message, level), level)
        
    def register_python_console(self, min_level=-10, max_level=10):
        """ Registers the python console. """
        self.register_printer_function(print, min_level, max_level)
    
    def reset(self):
        """ Resets the printers. """
        Printer.instance.clear()
    
def echo(message, level=0):
    Printer().echo(message, level)
    
def info(message):
    Printer().echo(message, Printer.INFO)
    
def debug(message):
    Printer().echo(message, Printer.DEBUG)

def warn(message):
    Printer().echo(message, Printer.WARNING)
    
def error(message):
    Printer().echo(message, Printer.ERROR)
    
def critical(message):
    Printer().echo(message, Printer.CRITICAL)
    
def init_echo(include_python=True, other_printers=[]):
    Printer().reset()
    if include_python:
        Printer().register_python_console()
    for printer in other_printers:
        Printer().register_printer_function(printer)