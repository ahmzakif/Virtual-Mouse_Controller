from virtual.mouse_controller import MouseController

def main():
    print("Virtual Mouse Control")
    print("---------------------")
    print("Press 'q' to quit")
    print("Press '+' to increase sensitivity")
    print("Press '-' to decrease sensitivity")
    
    mc = MouseController()
    mc.run()

if __name__ == "__main__":
    main()