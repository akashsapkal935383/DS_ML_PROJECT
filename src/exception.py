import sys
import os
from logger import logging 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def error_message_detail(error,error_details:sys):

        _,_,exe_tb=error_details.exc_info()

        file_name = exe_tb.tb_frame.f_code.co_filename

        error_mesaage="Error occured in python script name [{0}] line number is [{1}] error message[{2}]".format()

        file_name,exe_tb.tb_lineno,str(error)

        return error_mesaage



class CustomExection(Exception):

        def __init__(self,error_message,error_details:sys):
                super().__init__(error_message)


        def __str__(self):
                return self.error_message




# exception.py
class CustomException(Exception):
    def __init__(self, message, errors=None):
        super().__init__(message)
        self.errors = errors
            