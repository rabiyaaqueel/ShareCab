from flask import Flask, render_template, request, redirect, session, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
import redis, json, os, random, pickle, nltk, numpy as np
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import folium
from geopy.geocoders import Nominatim

# --------------------------------
# INIT
# --------------------------------
app = Flask(__name__)
app.secret_key = "supersecretkey"
geolocator = Nominatim(user_agent="ride_app")

# Redis connection
r = redis.Redis(host='redis-17009.c8.us-east-1-2.ec2.redns.redis-cloud.com', port=17009, password="p1YNWfO8U32w4fNu8BSfnho2TOYTZnKH", decode_responses=True)

# --------------------------------
# CHATBOT SETUP
# --------------------------------
lemmatizer = WordNetLemmatizer()
model = load_model("chatbot_model.h5")
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))
intents = json.loads(open("bot.json").read())

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(w.lower()) for w in sentence_words]
    return sentence_words

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    p = bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return classes[results[0][0]] if results else "noanswer"

def get_response(tag):
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "Sorry, I didn’t understand that."

@app.route("/chatbot", methods=["POST"])
def chatbot_response():
    user_msg = request.json.get("message")
    tag = predict_class(user_msg)
    response = get_response(tag)
    return jsonify({"response": response})

# --------------------------------
# HELPER FUNCTIONS FOR RIDE SHARING
# --------------------------------
def rget_json(key):
    v = r.get(key)
    return json.loads(v) if v else None

def rset_json(key, obj):
    r.set(key, json.dumps(obj))

def iter_rides():
    for key in r.scan_iter("ride:*"):
        if key == "ride:id:seq":
            continue
        yield key, rget_json(key)

def find_open_ride_for_driver_route(driver_email, pickup, drop):
    for key, ride in iter_rides():
        if (ride and ride.get('driver') == driver_email
                and ride.get('pickup') == pickup
                and ride.get('drop') == drop
                and ride.get('status') != 'full'):
            return key, ride
    return None, None

def find_latest_ride_for_driver(driver_email):
    latest = None
    latest_key = None
    for key, ride in iter_rides():
        if ride and ride.get('driver') == driver_email:
            if latest is None or int(ride.get('id', 0)) > int(latest.get('id', 0)):
                latest = ride
                latest_key = key
    return latest_key, latest

# --------------------------------
# ROUTES
# --------------------------------
# ----- HOME PAGE -----
@app.route('/')
def home():
    return render_template('index.html')

# ----- SIGNUP -----
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        data = request.form
        hashed_password = generate_password_hash(data['password'])

        role = data['role']
        email = data['email'].strip().lower()

        # prevent duplicates (optional)
        if r.exists(f"user:{email}") or r.exists(f"driver:{email}"):
            return "Email already registered. Please log in."

        user_info = {
            'name': data['name'],
            'email': email,
            'password': hashed_password,
            'gender': data['gender'],
            'role': role
        }

        if role == 'driver':
            # NOTE: we removed latitude/longitude; we store plain-text location
            capacity = int(data.get('capacity', '4') or 4)
            base_location = data.get('driver_location', '').strip()
            user_info['capacity'] = capacity
            user_info['base_location'] = base_location

            rset_json(f"driver:{email}", user_info)
            r.sadd("drivers", email)  # index of driver emails
        else:
            rset_json(f"user:{email}", user_info)
            r.sadd("users", email)    # index of user emails

        return redirect('/login')
    return render_template('signup.html')

# ----- LOGIN -----
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email'].strip().lower()
        password = request.form['password']

        # try driver record first, then user record
        u = rget_json(f"driver:{email}")
        if not u:
            u = rget_json(f"user:{email}")

        if u and check_password_hash(u['password'], password):
            session['user'] = u
            return redirect('/user_dashboard' if u['role'] == 'user' else '/driver_dashboard')

        return "Invalid credentials"
    return render_template('login.html')

# ----- USER DASHBOARD -----
@app.route('/user_dashboard')
def user_dashboard():
    return render_template('user_dashboard.html')

# ----- SHOW RIDES -----
@app.route('/show_rides', methods=['POST'])
def show_rides():
    pickup = request.form['pickup'].strip()
    drop = request.form['drop'].strip()

    # Convert pickup and drop into coordinates
    pickup_location = geolocator.geocode(pickup)
    drop_location  = geolocator.geocode(drop)
    if not pickup_location or not drop_location:
        return "Invalid pickup or drop location!"

    session['pickup'] = pickup
    session['drop'] = drop
    session['pickup_coords'] = (pickup_location.latitude, pickup_location.longitude)
    session['drop_coords'] = (drop_location.latitude, drop_location.longitude)

    # Build available drivers list with seats_left
    available_drivers = []
    for email in r.smembers("drivers"):
        d = rget_json(f"driver:{email}")
        if not d:
            continue

        # Does this driver already have a ride for this route that is not full?
        ride_key, ride = find_open_ride_for_driver_route(email, pickup, drop)
        if ride:
            seats_left = int(d['capacity']) - len(ride.get('passengers', []))
        else:
            seats_left = int(d['capacity'])

        if seats_left > 0:
            available_drivers.append({
                'name': d['name'],
                'gender': d['gender'],
                'email': d['email'],
                'seats_left': seats_left
            })

    # Other users on same route (names only)
    users_on_route = []
    for _, rr in iter_rides():
        if rr and rr.get('pickup') == pickup and rr.get('drop') == drop and rr.get('status') != 'full':
            users_on_route.extend([p['name'] for p in rr.get('passengers', [])])

    return render_template('available_rides.html', drivers=available_drivers, users=users_on_route)

# ----- BOOK RIDE -----
@app.route('/book_ride/<driver_email>', methods=['POST'])
def book_ride(driver_email):
    pickup = session.get('pickup')
    drop = session.get('drop')
    user_name = session['user']['name']

    driver = rget_json(f"driver:{driver_email}")
    if not driver:
        return "Driver not found"

    driver_capacity = int(driver['capacity'])

    # find existing open ride for this driver+route
    ride_key, ride = find_open_ride_for_driver_route(driver_email, pickup, drop)

    if ride:
        passengers = ride.get('passengers', [])
        passengers.append({'name': user_name, 'gender': session['user']['gender'], 'coords': session['pickup_coords']})
        ride['passengers'] = passengers

        if len(passengers) >= driver_capacity:
            ride['status'] = 'full'

        # even split fare among current passengers
        ride['per_head_fare'] = round(float(ride['fare']) / len(passengers), 2)
        rset_json(ride_key, ride)
    else:
        total_fare = 50  # demo flat fare
        new_id = r.incr("ride:id:seq")
        ride = {
            'id': int(new_id),
            'driver': driver_email,
            'pickup': pickup,
            'drop': drop,
            'passengers': [{'name': user_name, 'gender': session['user']['gender'], 'coords': session['pickup_coords']}],
            'capacity': driver_capacity,
            'fare': float(total_fare),
            'per_head_fare': float(total_fare),  # first passenger
            'status': 'pending'
        }
        rset_json(f"ride:{new_id}", ride)

    return redirect('/map/' + driver_email)

# ----- MAP -----
@app.route('/map/<driver_email>')
def show_map(driver_email):
    # Prefer ride for the current session route; else latest ride of driver
    pickup = session.get('pickup')
    drop = session.get('drop')
    ride_key, ride = (None, None)
    if pickup and drop:
        ride_key, ride = find_open_ride_for_driver_route(driver_email, pickup, drop)
    if not ride:
        ride_key, ride = find_latest_ride_for_driver(driver_email)
    if not ride:
        return "No ride found"

    driver = rget_json(f"driver:{driver_email}")
    if not driver:
        return "Driver not found"

    pickup_coords = session.get('pickup_coords')
    drop_coords = session.get('drop_coords')
    if not (pickup_coords and drop_coords):
        # if session missing, geocode from ride pickup/drop
        p_loc = geolocator.geocode(ride['pickup'])
        d_loc = geolocator.geocode(ride['drop'])
        if not p_loc or not d_loc:
            return "Could not geocode route"
        pickup_coords = (p_loc.latitude, p_loc.longitude)
        drop_coords = (d_loc.latitude, d_loc.longitude)

    # Get driver's base location coords
    base = driver.get('base_location', '')
    base_loc = geolocator.geocode(base) if base else None
    if not base_loc:
        return "Driver base location invalid!"
    driver_coords = (base_loc.latitude, base_loc.longitude)

    # Center map
    center_lat = (driver_coords[0] + pickup_coords[0] + drop_coords[0]) / 3
    center_lon = (driver_coords[1] + pickup_coords[1] + drop_coords[1]) / 3

    ride_map = folium.Map(location=[center_lat, center_lon], zoom_start=13, height=400)

    # Driver marker
    folium.Marker(driver_coords, popup=f"Driver: {driver['name']}", icon=folium.Icon(color='blue')).add_to(ride_map)

    # Pickup marker
    folium.Marker(pickup_coords, popup="Pickup Location", icon=folium.Icon(color='green')).add_to(ride_map)

    # Drop marker
    folium.Marker(drop_coords, popup="Drop Location", icon=folium.Icon(color='red')).add_to(ride_map)

    # Route polyline
    folium.PolyLine([driver_coords, pickup_coords, drop_coords], color='blue', weight=3, opacity=0.7).add_to(ride_map)

    # Save map to templates/map.html
    out_path = os.path.join('templates', 'map.html')
    ride_map.save(out_path)

    # --- Inject a floating Logout button into the generated Folium HTML (NEW) ---
    try:
        with open(out_path, 'r', encoding='utf-8') as f:
            html = f.read()
        inject = """
<style>
#logoutBtn{
 position:fixed; top:12px; right:12px; z-index:10000;
 background:#dc3545; color:#fff; border:none; padding:8px 12px;
 border-radius:8px; font-weight:600; cursor:pointer;
 box-shadow:0 2px 8px rgba(0,0,0,0.15);
}
#logoutBtn:hover{ opacity:0.9; }
</style>
<a id="logoutBtn" href="/logout">Logout</a>
"""
        # insert before closing </body>
        html = html.replace("</body>", inject + "\n</body>")
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(html)
    except Exception:
        # if anything goes wrong, we still render the map
        pass

    return render_template('map.html')

# ----- DRIVER DASHBOARD -----
@app.route('/driver_dashboard')
# ----- DRIVER DASHBOARD -----
@app.route('/driver_dashboard')
def driver_dashboard():
    driver_email = session['user']['email']
    pending_rides, confirmed_rides, full_rides = [], [], []
    for _, rr in iter_rides():
        if rr and rr.get('driver') == driver_email:
            if rr.get('status') == 'pending':
                pending_rides.append(rr)
            elif rr.get('status') == 'confirmed':
                confirmed_rides.append(rr)
            elif rr.get('status') == 'full':
                full_rides.append(rr)
    return render_template('driver_dashboard.html',
                           rides=pending_rides,
                           confirmed_rides=confirmed_rides,
                           full_rides=full_rides)

# ----- ACCEPT RIDE -----
@app.route('/accept_ride/<int:ride_id>')
def accept_ride(ride_id):
    key = f"ride:{ride_id}"
    rr = rget_json(key)
    if not rr:
        return "Ride not found"

    # Update ride status
    rr['status'] = 'confirmed'
    rset_json(key, rr)

    # Optional: if you want to separate confirmed rides
    r.sadd("confirmed_rides", ride_id)

    # Redirect back to dashboard so ride disappears from "pending"
    return redirect('/driver_dashboard')

# ----- LOGOUT -----
@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect('/login')

if __name__ == '__main__':
    # pip install redis geopy folium werkzeug
    # Make sure a Redis server is running (e.g., `redis-server`)
    app.run(debug=True)


