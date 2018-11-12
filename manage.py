import sys
import gesture_recognition

def show_help():
    
    print("Help:")
    print("Arguments are needed to be given like this:")
    print("\t python manage.py [camera index] [operation]\n")
    print("Operations")
    print("\t-c\t Simple Calculator Using Sign Language.")
    print("\t-d\t ASL Digit Regontion.")
    print("\t-a\t ASL Alphabet Recognition")
    

if(len(sys.argv)>1):

    camera_index = sys.argv[1]
    operation = sys.argv[2]
    
    if(operation == "-c" or operation == "-d" or operation == "-a"):
        pass
    else:
        if(operation != "-h"):
            print("Sorry Wrong Arguments")
        show_help()
else:
    print("No arguments are given")
    show_help()