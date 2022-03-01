from time import time


class TimeTracker:
    def __init__(self):
        self.time_start = time()

    def print(self, current_element, max_element):
        if current_element == 0:
            return
        # estimate remaining time in seconds
        time_taken = time() - self.time_start
        time_per_element = time_taken / current_element
        total_time = int(max_element * time_per_element)

        print("\t\t{} / {}".format(self.format_time_to_string(int(time_taken)),
                                   self.format_time_to_string(total_time)),
              end='\r')
        if current_element == max_element:
            print("\n")
        return

    @staticmethod
    def format_time_to_string(seconds):
        (days, seconds) = divmod(seconds, 86400)
        (hours, seconds) = divmod(seconds, 3600)
        (minutes, seconds) = divmod(seconds, 60)
        t_string = '{:2d}d {:2d}h {:2d}m {:2d}s'.format(days,
                                                        hours,
                                                        minutes,
                                                        seconds)
        return t_string
