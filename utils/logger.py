import os
from datetime import datetime


class Logger:

    def __init__(self, save_dir="outputs"):

        os.makedirs(save_dir, exist_ok=True)

        now = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.file = open(

            os.path.join(save_dir, f"log_{now}.txt"),

            "w"

        )

    def log(self, text):

        print(text)

        self.file.write(text + "\n")

        self.file.flush()

    def close(self):

        self.file.close()