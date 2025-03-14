import time

def clock(f):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        print(f"{f.__name__} took {end-start:.4f} seconds to execute")
        return result
    return wrapper

@clock
def sum():
    x = 0
    for i in range(100000):
        x += 1 * i
    return x

if __name__ == "__main__":
    sum()