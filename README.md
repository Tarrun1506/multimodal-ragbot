# 🌟 StarRAG Bot - Multimodal Document & Image RAG System

StarRAG Bot is a **powerful full-stack multimodal RAG (Retrieval-Augmented Generation) application** that allows you to chat with both **documents AND images** using advanced AI technologies. It combines local language models with cutting-edge OCR (Optical Character Recognition) to extract and understand text from any source - whether it's a PDF, Word document, or a screenshot.

![StarRAG Bot Screenshot](screenshot.png) <!-- You can add a screenshot of your app here -->

## 🎯 Key Features

### 📄 **Document Processing**
- **Multi-format Support**: Upload `.txt`, `.pdf`, `.docx` files
- **Intelligent Text Extraction**: Advanced text parsing and chunking
- **Vector Search**: FAISS-powered similarity search for accurate retrieval

### 🖼️ **Image Processing & OCR**
- **Dual OCR Engines**: EasyOCR (primary) + Tesseract OCR (fallback)
- **Multi-format Images**: Support for `.png`, `.jpg`, `.jpeg`, `.gif`, `.bmp`, `.tiff`
- **Advanced Preprocessing**: Image enhancement for better text extraction
- **Smart Text Recognition**: Handles screenshots, scanned documents, photos of text

### 💬 **Intelligent Chat Interface**
- **Context-Aware Responses**: Ask questions about both documents and images
- **Conversation Memory**: Persistent chat history with MongoDB
- **Real-time Processing**: Instant responses with loading indicators
- **Multi-source Queries**: Combine information from documents and images

### 🔒 **Privacy & Security**
- **100% Local Processing**: No data sent to external APIs
- **Offline OCR**: Complete text extraction without internet
- **Local Vector Store**: All embeddings stored locally with FAISS
- **Secure Storage**: MongoDB for conversation and document management

## 🛠️ Tech Stack

### **Frontend**
- **React 18** with modern hooks and context
- **Vite** for fast development and building
- **Tailwind CSS** for responsive, modern UI
- **Lucide React** for beautiful icons

### **Backend**
- **Python 3.11+** with Flask web framework
- **LangChain** for document processing and text splitting
- **Sentence Transformers** for text embeddings
- **FAISS** for efficient vector similarity search
- **MongoDB** for persistent storage

### **AI & ML Technologies**
- **EasyOCR** - Advanced neural OCR engine
- **Tesseract OCR** - Traditional OCR with preprocessing
- **OpenCV** - Image preprocessing and enhancement
- **PIL/Pillow** - Image manipulation and format support
- **Ollama** - Local language model inference

### **Document Processing**
- **PyPDF2** - PDF text extraction
- **python-docx** - Word document processing
- **RecursiveCharacterTextSplitter** - Intelligent text chunking

## 📋 Prerequisites

Before you begin, ensure you have the following installed:

### **Required Software**
- **[Node.js](https://nodejs.org/)** (v18 or later) - For frontend development
- **[Python](https://www.python.org/downloads/)** (v3.11 or later) - For backend and AI processing
- **[MongoDB](https://www.mongodb.com/try/download/community)** - For data persistence
- **[Ollama](https://ollama.com/)** - For local language model inference

### **Optional (for better OCR)**
- **[Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)** - Enhanced text extraction (see `INSTALL_TESSERACT.md`)

## 🚀 Quick Start

### **1. Clone & Setup**
```bash
git clone <your-repository-url>
cd local-rag-bot
```

### **2. Backend Setup**
```bash
cd backend
pip install -r requirements.txt
```

### **3. Frontend Setup**
```bash
cd ../frontend
npm install
```

### **4. Start MongoDB**
```bash
# Windows (if installed as service)
net start MongoDB

# macOS/Linux
sudo systemctl start mongod

# Or run manually
mongod
```

### **5. Setup Ollama & Model**
```bash
# Start Ollama service
ollama serve

# In another terminal, pull the model
ollama pull gemma3:1b
```

### **6. Run the Application**

**Option A: Run Both Servers Simultaneously**
```bash
# From project root
npm start
```

**Option B: Run Separately**
```bash
# Terminal 1: Backend
cd backend
python app.py

# Terminal 2: Frontend
cd frontend
npm run dev
```

## 🌐 Access Your Application

- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:5000
- **Health Check**: http://localhost:5000/api/health

## 📱 How to Use

### **Upload Documents & Images**
1. Click the **upload button** in the header
2. Select files:
   - **Documents**: `.txt`, `.pdf`, `.docx`
   - **Images**: `.png`, `.jpg`, `.jpeg`, `.gif`, `.bmp`, `.tiff`
3. Files are automatically processed and indexed

### **Chat with Your Content**
1. Type questions about your uploaded content
2. Ask about specific documents or images
3. Combine information from multiple sources

### **Example Queries**
```
"What does the chart in the uploaded image show?"
"Summarize the main points from the PDF document"
"Extract contact information from the business card image"
"What are the key findings in the research paper?"
"Compare the data in the spreadsheet screenshot with the report"
```

## 📁 Project Structure

```
local-rag-bot/
├── 📂 backend/                    # Python Flask Backend
│   ├── 🐍 app.py                  # Main Flask application with RAG pipeline
│   ├── 📄 requirements.txt        # Python dependencies
│   └── 📁 uploads/               # Temporary file storage
│
├── 📂 frontend/                   # React Frontend
│   ├── 📂 src/
│   │   ├── 📂 components/         # React components
│   │   │   ├── ChatArea.jsx       # Chat interface
│   │   │   ├── Header.jsx         # File upload & navigation
│   │   │   ├── InputArea.jsx      # Message input
│   │   │   └── Sidebar.jsx        # Conversation history
│   │   ├── StarRAGBot.jsx         # Main application component
│   │   └── App.jsx               # Root component
│   ├── 📄 package.json           # Frontend dependencies
│   └── 📄 vite.config.js         # Vite configuration
│
├── 📄 package.json               # Root package for concurrent scripts
├── 📄 README.md                  # This comprehensive guide
├── 📄 IMAGE_PROCESSING_GUIDE.md  # Detailed image processing guide
├── 📄 INSTALL_TESSERACT.md       # Tesseract installation guide
├── 🐍 install_image_deps.py      # Automated dependency installer
└── 🧪 test_image_processing.py   # OCR functionality test script
```

## 🔧 Configuration

### **Backend Configuration**
- **MongoDB**: `mongodb://localhost:27017/`
- **Database**: `starrag_bot`
- **Collections**: `documents`, `conversations`
- **Upload Directory**: `backend/uploads/`
- **Chunk Size**: 1000 characters with 200 overlap

### **OCR Configuration**
- **Primary OCR**: EasyOCR (English)
- **Fallback OCR**: Tesseract (if installed)
- **Image Preprocessing**: OpenCV with noise reduction and thresholding
- **Supported Languages**: English (expandable)

### **Model Configuration**
- **Embedding Model**: `all-MiniLM-L6-v2`
- **LLM**: `gemma3:1b` via Ollama
- **Vector Store**: FAISS IndexFlatL2
- **Similarity Threshold**: 0.3 (configurable)

## 🎯 Advanced Features

### **Image Processing Pipeline**
1. **File Upload** → Image saved temporarily
2. **OCR Processing** → Text extraction with dual engines
3. **Text Preprocessing** → Cleaning and formatting
4. **Chunking** → Intelligent text splitting
5. **Embedding** → Vector representation generation
6. **Indexing** → FAISS vector store update
7. **Storage** → MongoDB document persistence

### **Query Processing**
1. **User Query** → Natural language input
2. **Embedding** → Query vector generation
3. **Similarity Search** → FAISS nearest neighbors
4. **Context Retrieval** → Relevant chunks extraction
5. **LLM Generation** → Ollama response generation
6. **Response** → Formatted answer with sources

### **Conversation Management**
- **Session Tracking** → Unique conversation IDs
- **History Persistence** → MongoDB storage
- **Context Awareness** → Previous messages consideration
- **Multi-turn Dialogue** → Coherent conversation flow

## 🔍 Troubleshooting

### **Common Issues**

**1. MongoDB Connection Failed**
```bash
# Start MongoDB service
net start MongoDB  # Windows
sudo systemctl start mongod  # Linux
brew services start mongodb-community  # macOS
```

**2. Ollama Model Not Found**
```bash
# Check available models
ollama list

# Pull required model
ollama pull gemma3:1b
```

**3. OCR Not Working**
- Install Tesseract OCR (see `INSTALL_TESSERACT.md`)
- EasyOCR works without additional setup
- Check image quality and format

**4. Port Already in Use**
```bash
# Kill process on port 5000
netstat -ano | findstr :5000  # Windows
lsof -ti:5000 | xargs kill  # macOS/Linux
```

### **Performance Optimization**

**For Better OCR Performance:**
- Use high-resolution images (300+ DPI)
- Ensure good contrast (dark text on light background)
- Avoid blurry or distorted images
- Use supported image formats

**For Better RAG Performance:**
- Adjust chunk size based on document type
- Fine-tune similarity threshold
- Use GPU acceleration if available
- Optimize MongoDB indexes

## 📊 System Requirements

### **Minimum Requirements**
- **CPU**: 4 cores, 2.0 GHz
- **RAM**: 8 GB
- **Storage**: 5 GB free space
- **OS**: Windows 10/11, macOS 10.15+, Ubuntu 18.04+

### **Recommended Requirements**
- **CPU**: 8 cores, 3.0 GHz
- **RAM**: 16 GB
- **GPU**: CUDA-compatible (optional, for faster processing)
- **Storage**: 10 GB free space
- **Network**: Internet for initial model downloads

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### **Development Setup**
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Ollama** for local LLM inference
- **EasyOCR** for advanced text recognition
- **LangChain** for RAG pipeline components
- **FAISS** for efficient vector search
- **React** and **Tailwind CSS** for the beautiful UI

---

**🎉 Your multimodal RAG bot is ready to process both documents and images!**

For detailed image processing setup, see `IMAGE_PROCESSING_GUIDE.md`
For Tesseract installation, see `INSTALL_TESSERACT.md`
