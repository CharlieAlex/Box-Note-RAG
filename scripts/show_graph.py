from app.graph import create_app
from app.io import show_graph

if __name__ == "__main__":
    app = create_app()
    show_graph(app)
