import logging

def setup_logger(output_path=''):
    
    # Configure Root Logger
    log_format = logging.Formatter('%(asctime)s | [%(filename)s] | %(levelname)s: %(message)s', '%H:%M:%S')
    
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Console Output
    console = logging.StreamHandler()
    console.setFormatter(log_format)
    root.addHandler(console)

    # File Output
  #  file_log = logging.FileHandler(output_path)
   # file_log.setFormatter(log_format)
   # root.addHandler(file_log)