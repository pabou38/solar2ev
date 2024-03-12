#!/usr/bin/pyton3

# good overview
# https://www.youtube.com/watch?v=MjHpMCIvwsY


##################################
# decorator modules
# handy "wrapper" around functions 
# to be defined BEFORE FUNCTION (to be decorated) DEFINITION
##################################

from time import time, sleep
import datetime

# only work for function, not ramdom piece of code
 
# function are 1st class citizen. can be use as argument, be returned
def dec_elapse(func_to_be_decorated):

    ########################
    # define what to do "around" the function
    #######################

    def wrapper(*args, **kwargs):

        #print("wrapper *arg" , args)   # tuple
        #print("wrapper **kwarg" , kwargs) # dict
        # parameters of function BEEING decorated

        # do something before and after function to be decorated
        start = time() # float value that represents the seconds since the epoch.

        x = func_to_be_decorated(*args, **kwargs)

        elapse = time() - start
        print("==> elapse from decorator: %0.1f sec %d mn" % (elapse, elapse/60))

        return(x) # make sure the returned value of the function to be decorated is available

    return(wrapper)


"""
start_time = datetime.datetime.now()
end_time = datetime.datetime.now()

time_diff = (end_time - start_time)
execution_time_sec = time_diff.total_seconds() 

from time import perf_counter


"""
# decorator with arguments
# https://stackoverflow.com/questions/5929107/decorators-with-parameters
# https://www.artima.com/weblogs/viewpost.jsp?thread=240845#decorator-functions-with-decorator-arguments

def dec_elapse_arg(argument):
    def dec(function_to_be_decorated):
            def wrapper(*args, **kwargs):
                print("decorator: ", argument) # not the argument of the function being decorated, but @dec(arg)
                start = time()
                x = function_to_be_decorated(*args, **kwargs)
                elapse = time() - start
                print("==> elapse from decorator: %0.1f sec %d mn" % (elapse, elapse/60))
                return(x)
            
            return(wrapper)
    return(dec)



 
if __name__ == "__main__":
    print ("testing decorator")

    # syntaxis sugar
    # equivalent of: function_to_be_decorated = dec_elapse(function_to_be_decorated)


    @dec_elapse
    def function_to_be_decorated(n:int, id="default"):
        print("start function being decorated. mandatory %d, optional %s" %(n, id))

        for i in range(n):
            sleep(1)

        return(i) # returned value of function to be decorated
    


    @dec_elapse_arg("decorator argument")
    def function_to_be_decorated1(n:int, id="default"):
        print("start function being decorated. mandatory %d, optional %s" %(n, id))   
        sleep(1)     
        return("toto") # returned value of function to be decorated


    ret = function_to_be_decorated(2)
    print("function being decorated returned", ret)

    #wrappper *arg (2,)
    #wrappper **kwarg {}

    ret = function_to_be_decorated(1, id = "id")
    print("function being decorated returned", ret)

    #wrappper *arg (1,)
    #wrappper **kwarg {'id': 'id'}

    ret = function_to_be_decorated1(5, id = "arg")
    print("function being decorated with decorator argument returned", ret)






