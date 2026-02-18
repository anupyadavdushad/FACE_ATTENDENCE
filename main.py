from src.registration.capture import data_collecter
from src.create_embedding import create_embeddings
from src.recognition.matcher import recognize_and_mark_attendance


def main():
    print("1. Register Student")
    print("2. Create Embeddings")
    print("3. Mark Attendance")

    choice = int(input("Enter choice: "))

    if choice == 1:
        name = input("Enter name: ")
        reg_no = input("Enter reg no: ")
        data_collecter(name, reg_no)

    elif choice == 2:
        create_embeddings()
        print("Embeddings created successfully.")

    elif choice == 3:
        recognize_and_mark_attendance()

    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
