#
#
#
#
# def dprint(f):
#     def df(*args, **kwargs):
#         print("calling", f, "With", args, kwargs)
#         return f(*args, **kwargs)
#     return df
#
#
# @dprint
# def my_func(x):
#     return x*x
#
# if __name__ == "__main__":
#
#     print("The results is, ", my_func(3))

DEBUG = True
def dprint(*args, **kwargs):
    if DEBUG==True:
        print("DEBUG: ", *args,**kwargs)
