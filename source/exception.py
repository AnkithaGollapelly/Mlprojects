import sys
import logging
def error_message_detail(error,error_details:sys):
                    _,_,ecx_tberror_details.exc_info()
                    file_name=exc_tb.tb_frame.f_code.co_filename
                    error_mesasge='error occured in python scripy name[{0}] line number [{1}] error message[{2}]'.format[{2}]
                    file_name,exc_tb.tb_lineno,str(error)
                    return error_mesasge
class CustomException(Exception):
                    def __init__(self,error_message,error_detail:sys):
                                        super().__init__(error_message)
                                        self.error_message=error_message_detail(error_message,error_detail=error_details)
                    def __str__(self):
                                        return self.error_message
'''if __name__=="__main__":
                    try:

                                        a=1/0
                    except Exception as e:
                                        #logging.info("divided by zero")
                                        raise CustomException(e,sys)'''


                                        