from app import app

if __name__ == '__main__':
    # You can modify this for production later
    app.run(host='0.0.0.0', port=5000, debug=True)
