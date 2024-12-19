from flask import Flask, jsonify, request

app = Flask(__name__)

# Exemple de données fictives
books = [
    {"id": 1, "title": "1984", "author": "George Orwell"},
    {"id": 2, "title": "Brave New World", "author": "Aldous Huxley"},
]

@app.route("/")
def home():
    return "Bienvenue sur la page d'accueil!"

@app.route("/api/v1/resource")
def resource():
    return {"message": "Voici une ressource."}

# Route pour récupérer tous les livres
@app.route('/books', methods=['GET'])
def get_books():
    return jsonify(books)  # Retourne les données au format JSON

# Route pour récupérer un livre par son ID
@app.route('/books/<int:book_id>', methods=['GET'])
def get_book(book_id):
    book = next((book for book in books if book["id"] == book_id), None)
    if book:
        return jsonify(book)
    return jsonify({"error": "Book not found"}), 404

# Route pour ajouter un nouveau livre
@app.route('/books', methods=['POST'])
def add_book():
    new_book = request.json
    books.append(new_book)
    return jsonify(new_book), 201

# Route pour supprimer un livre par son ID
@app.route('/books/<int:book_id>', methods=['DELETE'])
def delete_book(book_id):
    global books
    books = [book for book in books if book["id"] != book_id]
    return jsonify({"message": "Book deleted"}), 200

if __name__ == '__main__':
    app.run(port=5000, debug=True, use_reloader=False)
