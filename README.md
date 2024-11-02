# Upendo - A gentle Introduction to RAG with Gemini and Langchains

![Upendo Banner](https://raw.githubusercontent.com/ngesa254/upendo/main/public/banner.png)

A comprehensive implementation of Retrieval Augmented Generation (RAG) using Gemini API and LangChain, featuring a modern Angular frontend and Python backend.

[![GitHub stars](https://img.shields.io/github/stars/ngesa254/upendo)](https://github.com/ngesa254/upendo/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/ngesa254/upendo)](https://github.com/ngesa254/upendo/network)
[![GitHub issues](https://img.shields.io/github/issues/ngesa254/upendo)](https://github.com/ngesa254/upendo/issues)
[![GitHub license](https://img.shields.io/github/license/ngesa254/upendo)](https://github.com/ngesa254/upendo/blob/main/LICENSE)

## Project Overview

This project demonstrates the practical implementation of RAG systems with a user-friendly interface. It consists of two main components:

### Frontend (`angular-fe/`)
- Modern Gemini-style interface built with Angular 17
- Real-time chat interactions
- File upload and voice input capabilities
- Responsive design with Tailwind CSS

### Backend (`backend/`)
- RAG implementation using LangChain and Gemini API
- Document processing and retrieval system
- Vector database integration
- API endpoints for frontend communication

## Session Abstract

Developers will dive into the world of Retrieval Augmented Generation (RAG) with this comprehensive workshop. They will be guided through the process of using LangChain and the Gemini API to effectively retrieve and generate information.

### Key Takeaways
* **Gemini fundamentals:** Understand the capabilities, limitations, and different models within the Gemini family
* **LangChain mastery:** Learn how to use LangChain to load, store, and retrieve documents for information retrieval
* **RAG pipelines:** Build pipelines that combine retrieval and generation to answer complex queries
* **Gemini Vision integration:** Discover how to integrate Gemini's vision capabilities into your RAG applications
* **Q&A chatbots:** Create intelligent chatbots capable of interacting with stored documents
* **Vector DB:** Explore the role of Vector DB in building scalable RAG systems

## Quick Start Guide

### Prerequisites
- Node.js v18 or higher
- Python 3.8 or higher
- Git
- Google Cloud account with Gemini API access

### Frontend Setup (Angular)

```bash
# Clone the repository
git clone https://github.com/ngesa254/upendo.git
cd upendo/angular-fe

# Install dependencies
npm install

# Start development server
ng serve
```

Access the application at `http://localhost:4200`

### Backend Setup (Python)

```bash
# Navigate to backend directory
cd upendo/backend

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your Gemini API key

# Start the backend server
python main.py
```

## Demo Applications

1. **Llama Index Chatbot**
   - Built with Gemini and Streamlit
   - Deployed on Google App Engine
   - Real-time document interaction
   - [Live Demo](https://github.com/ngesa254/upendo) (Coming Soon)

2. **LangChains Chatbot**
   - Integrated with Gemini API
   - Streamlit interface
   - App Engine deployment
   - [Live Demo](https://github.com/ngesa254/upendo) (Coming Soon)

## Technical Stack

### Frontend
- Angular 17
- TypeScript
- Tailwind CSS
- RxJS

### Backend
- Python
- LangChain
- Gemini API
- Vector Database
- FastAPI/Flask

## Project Structure
```
Upendo/
├── .ipynb_checkpoints/
├── angular-fe/          # Angular frontend application
├── backend/            # Python RAG implementation
└── README.md
```

## Development Roadmap

- [x] Project setup and repository creation
- [x] Frontend UI implementation with Gemini design
- [ ] Backend RAG system implementation
- [ ] API integration between frontend and backend
- [ ] Vector DB setup and configuration
- [ ] Document processing pipeline
- [ ] Chat functionality with Gemini API
- [ ] Deployment configuration for GCP

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Support

If you're having any problems, please raise an issue at https://github.com/ngesa254/upendo/issues.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Ngesa N. Marvin** 
- [GitHub Profile](https://github.com/ngesa254)
- [LinkedIn Profile](https://www.linkedin.com/in/ngesa-marvin/)
- [Twitter Profile](https://twitter.com/MarvinNgesa)

## Acknowledgments

* Google Cloud & Gemini API team
* LangChain community
* Angular development team
* All contributors and supporters