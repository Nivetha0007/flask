from flask import Flask, request, jsonify, render_template, redirect, url_for
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Product data with 15 products per category
products = [
    # Student Category (15 products)
    {"id": 1, "name": "Budget Laptop", "category": "student", "subCategory": "electronics", "price": 499.99, "description": "Affordable laptop for students...", "image": "/static/images/budget_laptop.jpg", "tags": ["student", "tech", "study"]},
    {"id": 2, "name": "Graphing Calculator", "category": "student", "subCategory": "tools", "price": 99.99, "description": "Essential for math classes...", "image": "/static/images/graphing_calculator.jpg", "tags": ["student", "math", "tools"]},
    {"id": 3, "name": "Backpack Pro", "category": "student", "subCategory": "accessories", "price": 59.99, "description": "Durable student backpack...", "image": "/static/images/backpack_pro.jpg", "tags": ["student", "carry", "school"]},
    {"id": 4, "name": "Study Desk Lamp", "category": "student", "subCategory": "furniture", "price": 29.99, "description": "Bright LED lamp for studying...", "image": "/static/images/study_lamp.jpg", "tags": ["student", "light", "desk"]},
    {"id": 5, "name": "Notebook Set", "category": "student", "subCategory": "stationery", "price": 14.99, "description": "Pack of 5 notebooks...", "image": "/static/images/notebook_set.jpg", "tags": ["student", "notes", "write"]},
    {"id": 6, "name": "USB Flash Drive", "category": "student", "subCategory": "electronics", "price": 19.99, "description": "32GB storage for projects...", "image": "/static/images/usb_drive.jpg", "tags": ["student", "tech", "storage"]},
    {"id": 7, "name": "Planner 2025", "category": "student", "subCategory": "stationery", "price": 12.99, "description": "Organize your school year...", "image": "/static/images/planner_2025.jpg", "tags": ["student", "plan", "time"]},
    {"id": 8, "name": "Noise-Canceling Headphones", "category": "student", "subCategory": "electronics", "price": 79.99, "description": "Focus on studies...", "image": "/static/images/noise_headphones.jpg", "tags": ["student", "audio", "focus"]},
    {"id": 9, "name": "Textbook Organizer", "category": "student", "subCategory": "furniture", "price": 39.99, "description": "Keep books tidy...", "image": "/static/images/textbook_organizer.jpg", "tags": ["student", "storage", "books"]},
    {"id": 10, "name": "Portable Charger", "category": "student", "subCategory": "electronics", "price": 24.99, "description": "10,000mAh power bank...", "image": "/static/images/portable_charger.jpg", "tags": ["student", "tech", "power"]},
    {"id": 11, "name": "Highlighter Set", "category": "student", "subCategory": "stationery", "price": 9.99, "description": "6 vibrant colors...", "image": "/static/images/highlighter_set.jpg", "tags": ["student", "study", "write"]},
    {"id": 12, "name": "Student Desk Chair", "category": "student", "subCategory": "furniture", "price": 89.99, "description": "Ergonomic chair...", "image": "/static/images/desk_chair.jpg", "tags": ["student", "comfort", "seat"]},
    {"id": 13, "name": "Math Textbook", "category": "student", "subCategory": "books", "price": 49.99, "description": "Calculus essentials...", "image": "/static/images/math_textbook.jpg", "tags": ["student", "study", "math"]},
    {"id": 14, "name": "Laptop Stand", "category": "student", "subCategory": "accessories", "price": 34.99, "description": "Adjustable stand...", "image": "/static/images/laptop_stand.jpg", "tags": ["student", "tech", "ergonomics"]},
    {"id": 15, "name": "Water Bottle", "category": "student", "subCategory": "accessories", "price": 19.99, "description": "Insulated 500ml bottle...", "image": "/static/images/water_bottle.jpg", "tags": ["student", "hydration", "carry"]},

    # Tech Enthusiast Category (15 products)
    {"id": 16, "name": "Gaming PC", "category": "tech enthusiast", "subCategory": "electronics", "price": 1499.99, "description": "High-end gaming rig...", "image": "/static/images/gaming_pc.jpg", "tags": ["tech", "gaming", "power"]},
    {"id": 17, "name": "VR Headset", "category": "tech enthusiast", "subCategory": "electronics", "price": 399.99, "description": "Immersive VR experience...", "image": "/static/images/vr_headset.jpg", "tags": ["tech", "vr", "gaming"]},
    {"id": 18, "name": "Smartphone Pro", "category": "tech enthusiast", "subCategory": "electronics", "price": 999.99, "description": "Latest flagship phone...", "image": "/static/images/smartphone_pro.jpg", "tags": ["tech", "mobile", "photography"]},
    {"id": 19, "name": "Mechanical Keyboard", "category": "tech enthusiast", "subCategory": "accessories", "price": 129.99, "description": "RGB mechanical keys...", "image": "/static/images/mech_keyboard.jpg", "tags": ["tech", "typing", "gaming"]},
    {"id": 20, "name": "4K Monitor", "category": "tech enthusiast", "subCategory": "electronics", "price": 349.99, "description": "Ultra HD display...", "image": "/static/images/4k_monitor.jpg", "tags": ["tech", "display", "work"]},
    {"id": 21, "name": "Wireless Mouse", "category": "tech enthusiast", "subCategory": "accessories", "price": 59.99, "description": "Precision gaming mouse...", "image": "/static/images/wireless_mouse.jpg", "tags": ["tech", "gaming", "control"]},
    {"id": 22, "name": "Smart Home Hub", "category": "tech enthusiast", "subCategory": "electronics", "price": 149.99, "description": "Control your smart home...", "image": "/static/images/smart_hub.jpg", "tags": ["tech", "smart", "home"]},
    {"id": 23, "name": "Drone Kit", "category": "tech enthusiast", "subCategory": "electronics", "price": 299.99, "description": "4K camera drone...", "image": "/static/images/drone_kit.jpg", "tags": ["tech", "fly", "photography"]},
    {"id": 24, "name": "SSD Drive", "category": "tech enthusiast", "subCategory": "electronics", "price": 89.99, "description": "1TB fast storage...", "image": "/static/images/ssd_drive.jpg", "tags": ["tech", "storage", "speed"]},
    {"id": 25, "name": "Tech Magazine", "category": "tech enthusiast", "subCategory": "books", "price": 19.99, "description": "Latest tech trends...", "image": "/static/images/tech_magazine.jpg", "tags": ["tech", "reading", "news"]},
    {"id": 26, "name": "Smart Watch Ultra", "category": "tech enthusiast", "subCategory": "electronics", "price": 299.99, "description": "Advanced fitness tracker...", "image": "/static/images/smart_watch_ultra.jpg", "tags": ["tech", "fitness", "wearable"]},
    {"id": 27, "name": "Gaming Chair", "category": "tech enthusiast", "subCategory": "furniture", "price": 199.99, "description": "Ergonomic gaming seat...", "image": "/static/images/gaming_chair.jpg", "tags": ["tech", "comfort", "gaming"]},
    {"id": 28, "name": "Bluetooth Speaker", "category": "tech enthusiast", "subCategory": "electronics", "price": 79.99, "description": "Portable high-quality sound...", "image": "/static/images/bluetooth_speaker.jpg", "tags": ["tech", "audio", "music"]},
    {"id": 29, "name": "Coding Book", "category": "tech enthusiast", "subCategory": "books", "price": 34.99, "description": "Learn Python programming...", "image": "/static/images/coding_book.jpg", "tags": ["tech", "coding", "learn"]},
    {"id": 30, "name": "USB-C Hub", "category": "tech enthusiast", "subCategory": "accessories", "price": 49.99, "description": "Multi-port adapter...", "image": "/static/images/usb_c_hub.jpg", "tags": ["tech", "connectivity", "work"]},

    # Casual Category (15 products)
    {"id": 31, "name": "T-Shirt Pack", "category": "casual", "subCategory": "clothing", "price": 29.99, "description": "Pack of 3 comfy tees...", "image": "/static/images/tshirt_pack.jpg", "tags": ["casual", "fashion", "comfort"]},
    {"id": 32, "name": "Sneakers", "category": "casual", "subCategory": "footwear", "price": 69.99, "description": "Stylish everyday shoes...", "image": "/static/images/sneakers.jpg", "tags": ["casual", "fashion", "walk"]},
    {"id": 33, "name": "Hoodie", "category": "casual", "subCategory": "clothing", "price": 39.99, "description": "Cozy cotton hoodie...", "image": "/static/images/hoodie.jpg", "tags": ["casual", "fashion", "warm"]},
    {"id": 34, "name": "Board Game", "category": "casual", "subCategory": "entertainment", "price": 34.99, "description": "Fun for friends...", "image": "/static/images/board_game.jpg", "tags": ["casual", "fun", "games"]},
    {"id": 35, "name": "Coffee Mug", "category": "casual", "subCategory": "accessories", "price": 14.99, "description": "Funny quote mug...", "image": "/static/images/coffee_mug.jpg", "tags": ["casual", "drink", "home"]},
    {"id": 36, "name": "Wireless Earbuds", "category": "casual", "subCategory": "electronics", "price": 49.99, "description": "Affordable earbuds...", "image": "/static/images/wireless_earbuds_casual.jpg", "tags": ["casual", "audio", "music"]},
    {"id": 37, "name": "Casual Backpack", "category": "casual", "subCategory": "accessories", "price": 44.99, "description": "Lightweight carry bag...", "image": "/static/images/casual_backpack.jpg", "tags": ["casual", "carry", "travel"]},
    {"id": 38, "name": "Novel Bestseller", "category": "casual", "subCategory": "books", "price": 19.99, "description": "Relaxing fiction read...", "image": "/static/images/novel_bestseller.jpg", "tags": ["casual", "reading", "fiction"]},
    {"id": 39, "name": "Sunglasses", "category": "casual", "subCategory": "accessories", "price": 24.99, "description": "Trendy UV protection...", "image": "/static/images/sunglasses.jpg", "tags": ["casual", "fashion", "sun"]},
    {"id": 40, "name": "Yoga Mat", "category": "casual", "subCategory": "fitness", "price": 29.99, "description": "Non-slip exercise mat...", "image": "/static/images/yoga_mat.jpg", "tags": ["casual", "fitness", "health"]},
    {"id": 41, "name": "Portable Speaker", "category": "casual", "subCategory": "electronics", "price": 39.99, "description": "Compact sound...", "image": "/static/images/portable_speaker.jpg", "tags": ["casual", "audio", "party"]},
    {"id": 42, "name": "Denim Jeans", "category": "casual", "subCategory": "clothing", "price": 49.99, "description": "Classic blue jeans...", "image": "/static/images/denim_jeans.jpg", "tags": ["casual", "fashion", "comfort"]},
    {"id": 43, "name": "Puzzle Set", "category": "casual", "subCategory": "entertainment", "price": 24.99, "description": "1000-piece puzzle...", "image": "/static/images/puzzle_set.jpg", "tags": ["casual", "fun", "relax"]},
    {"id": 44, "name": "Cap", "category": "casual", "subCategory": "accessories", "price": 19.99, "description": "Stylish baseball cap...", "image": "/static/images/cap.jpg", "tags": ["casual", "fashion", "sun"]},
    {"id": 45, "name": "Cookbook Easy", "category": "casual", "subCategory": "books", "price": 22.99, "description": "Simple recipes...", "image": "/static/images/cookbook_easy.jpg", "tags": ["casual", "cooking", "food"]},

    # Teacher Category (15 products)
    {"id": 46, "name": "Whiteboard Markers", "category": "teacher", "subCategory": "stationery", "price": 14.99, "description": "Pack of 10 markers...", "image": "/static/images/whiteboard_markers.jpg", "tags": ["teacher", "teach", "write"]},
    {"id": 47, "name": "Teacher Planner", "category": "teacher", "subCategory": "stationery", "price": 19.99, "description": "Lesson plan organizer...", "image": "/static/images/teacher_planner.jpg", "tags": ["teacher", "plan", "time"]},
    {"id": 48, "name": "Projector", "category": "teacher", "subCategory": "electronics", "price": 299.99, "description": "HD classroom projector...", "image": "/static/images/projector.jpg", "tags": ["teacher", "tech", "present"]},
    {"id": 49, "name": "Desk Organizer", "category": "teacher", "subCategory": "furniture", "price": 34.99, "description": "Keep desk tidy...", "image": "/static/images/desk_organizer.jpg", "tags": ["teacher", "storage", "desk"]},
    {"id": 50, "name": "Educational Posters", "category": "teacher", "subCategory": "stationery", "price": 24.99, "description": "Set of 5 posters...", "image": "/static/images/edu_posters.jpg", "tags": ["teacher", "teach", "visual"]},
    {"id": 51, "name": "Laptop for Teachers", "category": "teacher", "subCategory": "electronics", "price": 699.99, "description": "Lightweight teaching laptop...", "image": "/static/images/teacher_laptop.jpg", "tags": ["teacher", "tech", "work"]},
    {"id": 52, "name": "Teacher Mug", "category": "teacher", "subCategory": "accessories", "price": 12.99, "description": "Inspirational teacher mug...", "image": "/static/images/teacher_mug.jpg", "tags": ["teacher", "drink", "motivation"]},
    {"id": 53, "name": "Bookshelf", "category": "teacher", "subCategory": "furniture", "price": 89.99, "description": "Classroom bookshelf...", "image": "/static/images/bookshelf.jpg", "tags": ["teacher", "storage", "books"]},
    {"id": 54, "name": "Laser Pointer", "category": "teacher", "subCategory": "tools", "price": 19.99, "description": "Presentation pointer...", "image": "/static/images/laser_pointer.jpg", "tags": ["teacher", "teach", "present"]},
    {"id": 55, "name": "Grading Software", "category": "teacher", "subCategory": "electronics", "price": 49.99, "description": "Digital scanner tool...", "image": "/static/images/scanner_software.jpg", "tags": ["teacher", "tech", "grade"]},
    {"id": 56, "name": "Teacher Tote Bag", "category": "teacher", "subCategory": "accessories", "price": 29.99, "description": "Spacious carry bag...", "image": "/static/images/teacher_tote.jpg", "tags": ["teacher", "carry", "work"]},
    {"id": 57, "name": "History Book", "category": "teacher", "subCategory": "books", "price": 39.99, "description": "Comprehensive history text...", "image": "/static/images/history_book.jpg", "tags": ["teacher", "reading", "history"]},
    {"id": 58, "name": "Classroom Clock", "category": "teacher", "subCategory": "furniture", "price": 24.99, "description": "Large wall clock...", "image": "/static/images/classroom_clock.jpg", "tags": ["teacher", "time", "decor"]},
    {"id": 59, "name": "Pen Set", "category": "teacher", "subCategory": "stationery", "price": 9.99, "description": "Pack of 12 pens...", "image": "/static/images/pen_set.jpg", "tags": ["teacher", "write", "teach"]},
    {"id": 60, "name": "Smart Board", "category": "teacher", "subCategory": "electronics", "price": 999.99, "description": "Interactive teaching board...", "image": "/static/images/smart_board.jpg", "tags": ["teacher", "tech", "interactive"]},

    # Business Category (15 products)
    {"id": 61, "name": "Business Laptop", "category": "business", "subCategory": "electronics", "price": 1299.99, "description": "High-performance work laptop...", "image": "/static/images/business_laptop.jpg", "tags": ["business", "tech", "work"]},
    {"id": 62, "name": "Briefcase", "category": "business", "subCategory": "accessories", "price": 89.99, "description": "Leather business briefcase...", "image": "/static/images/briefcase.jpg", "tags": ["business", "carry", "professional"]},
    {"id": 63, "name": "Office Chair", "category": "business", "subCategory": "furniture", "price": 199.99, "description": "Ergonomic office seat...", "image": "/static/images/office_chair.jpg", "tags": ["business", "comfort", "work"]},
    {"id": 64, "name": "Business Suit", "category": "business", "subCategory": "clothing", "price": 249.99, "description": "Tailored suit...", "image": "/static/images/business_suit.jpg", "tags": ["business", "fashion", "professional"]},
    {"id": 65, "name": "Smart Pen", "category": "business", "subCategory": "stationery", "price": 79.99, "description": "Digital note-taking pen...", "image": "/static/images/smart_pen.jpg", "tags": ["business", "tech", "write"]},
    {"id": 66, "name": "Desk Phone", "category": "business", "subCategory": "electronics", "price": 59.99, "description": "VoIP office phone...", "image": "/static/images/desk_phone.jpg", "tags": ["business", "tech", "communication"]},
    {"id": 67, "name": "Business Planner", "category": "business", "subCategory": "stationery", "price": 24.99, "description": "Professional organizer...", "image": "/static/images/business_planner.jpg", "tags": ["business", "plan", "time"]},
    {"id": 68, "name": "Monitor Stand", "category": "business", "subCategory": "furniture", "price": 39.99, "description": "Adjustable stand...", "image": "/static/images/monitor_stand.jpg", "tags": ["business", "tech", "ergonomics"]},
    {"id": 69, "name": "Business Book", "category": "business", "subCategory": "books", "price": 29.99, "description": "Leadership insights...", "image": "/static/images/business_book.jpg", "tags": ["business", "reading", "leadership"]},
    {"id": 70, "name": "Wireless Presenter", "category": "business", "subCategory": "electronics", "price": 34.99, "description": "Presentation clicker...", "image": "/static/images/wireless_presenter.jpg", "tags": ["business", "tech", "present"]},
    {"id": 71, "name": "Dress Shoes", "category": "business", "subCategory": "footwear", "price": 99.99, "description": "Polished leather shoes...", "image": "/static/images/dress_shoes.jpg", "tags": ["business", "fashion", "professional"]},
    {"id": 72, "name": "Office Desk", "category": "business", "subCategory": "furniture", "price": 299.99, "description": "Spacious work desk...", "image": "/static/images/office_desk.jpg", "tags": ["business", "work", "furniture"]},
    {"id": 73, "name": "Business Card Holder", "category": "business", "subCategory": "accessories", "price": 19.99, "description": "Metal card case...", "image": "/static/images/card_holder.jpg", "tags": ["business", "professional", "carry"]},
    {"id": 74, "name": "Noise-Canceling Mic", "category": "business", "subCategory": "electronics", "price": 69.99, "description": "Clear audio for calls...", "image": "/static/images/noise_mic.jpg", "tags": ["business", "tech", "communication"]},
    {"id": 75, "name": "Portfolio Folder", "category": "business", "subCategory": "stationery", "price": 24.99, "description": "Leather document holder...", "image": "/static/images/portfolio_folder.jpg", "tags": ["business", "work", "organize"]}
]

# Dynamic user management
user_product_matrix = {}  # {email: [views]}
user_cart = {}  # {email: [{id, quantity}]}
user_types = {}  # {email: type}

# Routes
@app.route('/')
def login():
    return render_template('login.html')

@app.route('/main')
def main():
    user = request.args.get('user')
    if not user:
        return redirect(url_for('login'))
    
    if user not in user_types:
        user_types[user] = 'casual'  # Default type
        user_product_matrix[user] = np.zeros(len(products))
        user_cart[user] = []
    
    return render_template('main.html', user=user)

@app.route('/cart')
def cart():
    user = request.args.get('user')
    if not user or user not in user_cart:
        return redirect(url_for('login'))
    
    cart_items = user_cart.get(user, [])
    total = sum(products[next(i for i, p in enumerate(products) if p['id'] == item['id'])]['price'] * item['quantity'] for item in cart_items)
    return render_template('cart.html', user=user, cart_items=cart_items, products=products, total=total)

@app.route('/products')
def get_products():
    category = request.args.get('category', 'all')
    if category == 'all':
        return jsonify(products)
    return jsonify([p for p in products if p['category'] == category])

@app.route('/search')
def search():
    query = request.args.get('query', '').lower()
    if not query:
        return jsonify(products)
    results = [p for p in products if query in p['name'].lower() or query in p['category'].lower() or any(query in tag.lower() for tag in p['tags'])]
    return jsonify(results)

@app.route('/product/<int:product_id>', methods=['POST'])
def view_product(product_id):
    user = request.args.get('user')
    if user not in user_product_matrix:
        user_product_matrix[user] = np.zeros(len(products))
    product_idx = next(i for i, p in enumerate(products) if p['id'] == product_id)
    user_product_matrix[user][product_idx] = 1
    
    product = products[product_idx]
    similarities = [(p, sum(1 for t in p['tags'] if t in product['tags']) / len(p['tags'])) for p in products if p['id'] != product_id]
    similar = [p for p, _ in sorted(similarities, key=lambda x: x[1], reverse=True)][:4]
    
    return jsonify({'product': product, 'similar': similar})

@app.route('/cart/add', methods=['POST'])
def add_to_cart():
    user = request.args.get('user')
    product_id = int(request.args.get('product_id'))
    if user not in user_cart:
        user_cart[user] = []
    cart = user_cart[user]
    existing = next((item for item in cart if item['id'] == product_id), None)
    if existing:
        existing['quantity'] += 1
    else:
        cart.append({'id': product_id, 'quantity': 1})
    return '', 204

@app.route('/cart/count')
def cart_count():
    user = request.args.get('user')
    count = sum(item['quantity'] for item in user_cart.get(user, []))
    return jsonify({'count': count})

@app.route('/cart/items')
def cart_items():
    user = request.args.get('user')
    return jsonify(user_cart.get(user, []))

@app.route('/user/update', methods=['POST'])
def update_user_type():
    user = request.args.get('user')
    user_type = request.args.get('type')
    user_types[user] = user_type
    return '', 204

@app.route('/recommend')
def recommend():
    user = request.args.get('user')
    if user not in user_product_matrix:
        user_product_matrix[user] = np.zeros(len(products))
    
    user_type = user_types.get(user, 'casual')
    same_type_users = [u for u, t in user_types.items() if t == user_type and u != user]
    
    # If no other users of the same type, use tag-based fallback
    if not same_type_users:
        preferences = {
            'student': ['student', 'study', 'tech'],
            'tech enthusiast': ['tech', 'gaming', 'electronics'],
            'casual': ['casual', 'fashion', 'comfort'],
            'teacher': ['teacher', 'teach', 'education'],
            'business': ['business', 'work', 'professional']
        }[user_type]
        scores = []
        for i, product in enumerate(products):
            if user_product_matrix[user][i] == 0:  # Not viewed by current user
                pref_score = sum(1 for tag in product['tags'] if tag in preferences) / len(product['tags'])
                scores.append((product, pref_score))
        recommended = [p for p, _ in sorted(scores, key=lambda x: x[1], reverse=True)][:4]
        return jsonify(recommended)
    
    # Collaborative filtering with same-type users
    all_users = list(user_product_matrix.keys())
    matrix = np.array([user_product_matrix[u] for u in all_users])
    similarity_matrix = cosine_similarity(matrix)
    user_idx = all_users.index(user)
    user_vector = user_product_matrix[user]
    
    # Aggregate views from same-type users
    type_indices = [all_users.index(u) for u in same_type_users]
    type_views = matrix[type_indices].sum(axis=0)
    
    scores = []
    for i, product in enumerate(products):
        if user_vector[i] == 0:  # Not viewed by current user
            collab_score = type_views[i] / len(same_type_users) if same_type_users else 0
            similarity_score = sum(similarity_matrix[user_idx, j] for j in type_indices) / len(type_indices) if type_indices else 0
            scores.append((product, collab_score * 0.7 + similarity_score * 0.3))
    
    recommended = [p for p, _ in sorted(scores, key=lambda x: x[1], reverse=True)][:4]
    return jsonify(recommended)

if __name__ == '__main__':
    app.run(debug=True)