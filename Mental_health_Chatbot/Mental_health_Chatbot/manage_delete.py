# manage_delete.py
# Admin Tool for Deleting User, Chat, or Message by ID

from database_setup import app, db, User, Chat, Message

def delete_user(user_id):
    user = User.query.get(user_id)
    if not user:
        print("User not found.")
        return

    # Delete all chats & messages from this user
    chats = Chat.query.filter_by(user_id=user.id).all()
    for chat in chats:
        Message.query.filter_by(chat_id=chat.id).delete()
        db.session.delete(chat)

    db.session.delete(user)
    db.session.commit()
    print(f"User {user_id} deleted along with all their chats & messages.")


def delete_chat(chat_id):
    chat = Chat.query.get(chat_id)
    if not chat:
        print("Chat not found.")
        return

    Message.query.filter_by(chat_id=chat.id).delete()
    db.session.delete(chat)
    db.session.commit()
    print(f"Chat {chat_id} deleted successfully.")


def delete_message(message_id):
    msg = Message.query.get(message_id)
    if not msg:
        print("Message not found.")
        return

    db.session.delete(msg)
    db.session.commit()
    print(f"Message {message_id} deleted successfully.")


def menu():
    print("\n====== DELETE TOOL ======")
    print("1. Delete User by ID")
    print("2. Delete Chat by ID")
    print("3. Delete Message by ID")
    print("0. Exit")
    print("==========================")

    choice = input("Select an option: ")

    with app.app_context():
        if choice == "1":
            uid = input("Enter User ID to delete: ")
            delete_user(int(uid))

        elif choice == "2":
            cid = input("Enter Chat ID to delete: ")
            delete_chat(int(cid))

        elif choice == "3":
            mid = input("Enter Message ID to delete: ")
            delete_message(int(mid))

        elif choice == "0":
            print("Exiting...")
            return

        else:
            print("Invalid option.")

        # Restart menu after action
        menu()


if __name__ == "__main__":
    menu()
