import React, { useRef, useState } from 'react';
import { Send, Paperclip, X, FileText, Image, File, Trash2, Music } from 'lucide-react';

const InputArea = ({
  inputMessage,
  setInputMessage,
  handleSendMessage,
  isLoading,
  onUploadFile,
  documents,
  onDeleteDocument
}) => {
  const [showFileDropdown, setShowFileDropdown] = useState(false);
  const fileInputRef = useRef(null);

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleFileSelect = (e) => {
    onUploadFile(e);
    setShowFileDropdown(false);
  };

  const handleDeleteDocument = async (docId, filename) => {
    if (window.confirm(`Are you sure you want to delete "${filename}"?`)) {
      try {
        await onDeleteDocument(docId);
      } catch (error) {
        console.error('Error deleting document:', error);
        alert('Failed to delete document. Please try again.');
      }
    }
  };

  const getFileIcon = (filename) => {
    const ext = filename.split('.').pop()?.toLowerCase();
    if (['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'].includes(ext)) {
      return <Image size={16} />;
    } else if (['pdf'].includes(ext)) {
      return <FileText size={16} />;
    } else if (['mp3', 'wav', 'm4a', 'ogg'].includes(ext)) {
      return <Music size={16} />;
    }
    return <File size={16} />;
  };

  // ChatGPT-like dark theme styles
  const inputAreaStyles = {
    borderTop: '1px solid #4a4b57',
    backgroundColor: '#343541',
    padding: '16px',
    position: 'relative',
  };

  const containerStyles = {
    maxWidth: '768px',
    margin: '0 auto',
    position: 'relative',
  };

  const inputWrapperStyles = {
    position: 'relative',
    marginBottom: '8px',
  };

  const textareaStyles = {
    width: '100%',
    padding: '16px 140px 16px 16px',
    border: '1px solid #4a4b57',
    borderRadius: '12px',
    resize: 'none',
    fontSize: '16px',
    lineHeight: '1.5',
    color: '#ffffff',
    backgroundColor: '#40414f',
    outline: 'none',
    transition: 'border-color 0.2s, box-shadow 0.2s',
    minHeight: '56px',
    maxHeight: '200px',
  };

  const textareaFocusStyles = {
    borderColor: '#19c37d',
    boxShadow: '0 0 0 2px rgba(25, 195, 125, 0.1)',
  };

  const actionButtonsStyles = {
    position: 'absolute',
    right: '12px',
    bottom: '12px',
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
  };

  const iconButtonStyles = {
    padding: '8px',
    backgroundColor: 'transparent',
    border: 'none',
    borderRadius: '6px',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    transition: 'background-color 0.2s, color 0.2s',
    color: '#acacbe',
  };
  const iconButtonHoverStyles = {
    backgroundColor: '#4a4b57',
    color: '#ffffff',
  };

  const sendButtonStyles = {
    padding: '8px',
    backgroundColor: '#19c37d',
    border: 'none',
    borderRadius: '6px',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    transition: 'background-color 0.2s, transform 0.1s',
    color: '#ffffff',
  };

  const sendButtonHoverStyles = {
    backgroundColor: '#16a369',
    transform: 'scale(1.05)',
  };

  const sendButtonDisabledStyles = {
    backgroundColor: '#4a4b57',
    cursor: 'not-allowed',
    transform: 'none',
    color: '#6b7280',
  };

  const fileDropdownStyles = {
    position: 'absolute',
    bottom: '100%',
    right: '0',
    marginBottom: '8px',
    width: '280px',
    backgroundColor: '#40414f',
    border: '1px solid #4a4b57',
    borderRadius: '12px',
    boxShadow: '0 10px 25px rgba(0, 0, 0, 0.3)',
    zIndex: 50,
    padding: '16px',
  };

  const dropdownHeaderStyles = {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '12px',
    paddingBottom: '8px',
    borderBottom: '1px solid #4a4b57',
  };

  const dropdownTitleStyles = {
    fontSize: '14px',
    fontWeight: 600,
    color: '#ffffff',
  };

  const documentsListStyles = {
    maxHeight: '200px',
    overflowY: 'auto',
    marginBottom: '12px',
  };

  const documentItemStyles = {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    padding: '8px',
    borderRadius: '6px',
    marginBottom: '4px',
    transition: 'background-color 0.2s',
    color: '#acacbe',
    position: 'relative',
  };

  const documentItemHoverStyles = {
    backgroundColor: '#4a4b57',
  };

  const deleteButtonStyles = {
    opacity: 0,
    background: 'none',
    border: 'none',
    color: '#8e8ea0',
    cursor: 'pointer',
    padding: '4px',
    borderRadius: '4px',
    transition: 'all 0.2s',
    marginLeft: 'auto',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  };

  const deleteButtonHoverStyles = {
    color: '#ef4444',
    backgroundColor: 'rgba(239, 68, 68, 0.1)',
  };

  const deleteButtonVisibleStyles = {
    opacity: 1,
  };

  const addFilesButtonStyles = {
    width: '100%',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '8px',
    fontSize: '14px',
    backgroundColor: '#19c37d',
    color: '#ffffff',
    border: 'none',
    padding: '10px 16px',
    borderRadius: '8px',
    cursor: 'pointer',
    transition: 'background-color 0.2s',
  };

  const addFilesButtonHoverStyles = {
    backgroundColor: '#16a369',
  };

  const helperTextStyles = {
    textAlign: 'center',
    fontSize: '12px',
    color: '#acacbe',
    marginTop: '8px',
  };

  return (
    <div style={inputAreaStyles}>
      <div style={containerStyles}>
        <div style={inputWrapperStyles}>
          
          <textarea
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask StarRAG anything..."
            style={textareaStyles}
            onFocus={(e) => Object.assign(e.target.style, textareaFocusStyles)}
            onBlur={(e) => {
              e.target.style.borderColor = '#4a4b57';
              e.target.style.boxShadow = 'none';
            }}
            rows="1"
            disabled={isLoading}
          />
          
          <div style={actionButtonsStyles}>

            {/* File Upload Button */}
            <div style={{position: 'relative'}}>
              <button
                onClick={() => setShowFileDropdown(!showFileDropdown)}
                style={iconButtonStyles}
                onMouseEnter={(e) => Object.assign(e.target.style, iconButtonHoverStyles)}
                onMouseLeave={(e) => {
                  e.target.style.backgroundColor = 'transparent';
                  e.target.style.color = '#acacbe';
                }}
                title="Attach files"
                disabled={isLoading}
              >
                <Paperclip size={20} />
              </button>

              {showFileDropdown && (
                <div style={fileDropdownStyles}>
                  <div style={dropdownHeaderStyles}>
                    <h3 style={dropdownTitleStyles}>Attach Files</h3>
                    <button
                      onClick={() => setShowFileDropdown(false)}
                      style={iconButtonStyles}
                      onMouseEnter={(e) => Object.assign(e.target.style, iconButtonHoverStyles)}
                      onMouseLeave={(e) => {
                        e.target.style.backgroundColor = 'transparent';
                        e.target.style.color = '#acacbe';
                      }}
                    >
                      <X size={16} />
                    </button>
                  </div>

                  {documents.length > 0 && (
                    <>
                      <div style={{fontSize: '12px', color: '#acacbe', marginBottom: '8px'}}>
                        Recently uploaded:
                      </div>
                      <div style={documentsListStyles}>
                        {documents.slice(-5).map(doc => (
                          <div
                            key={doc.id}
                            style={documentItemStyles}
                            onMouseEnter={(e) => {
                              Object.assign(e.target.style, documentItemHoverStyles);
                              const deleteBtn = e.target.querySelector('.delete-btn');
                              if (deleteBtn) {
                                Object.assign(deleteBtn.style, deleteButtonVisibleStyles);
                              }
                            }}
                            onMouseLeave={(e) => {
                              e.target.style.backgroundColor = 'transparent';
                              const deleteBtn = e.target.querySelector('.delete-btn');
                              if (deleteBtn) {
                                deleteBtn.style.opacity = '0';
                              }
                            }}
                          >
                            {getFileIcon(doc.filename)}
                            <span style={{fontSize: '14px', flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap'}}>
                              {doc.filename}
                            </span>
                            <button
                              className="delete-btn"
                              onClick={(e) => {
                                e.stopPropagation();
                                handleDeleteDocument(doc.id, doc.filename);
                              }}
                              style={deleteButtonStyles}
                              onMouseEnter={(e) => {
                                e.stopPropagation();
                                Object.assign(e.target.style, deleteButtonHoverStyles);
                              }}
                              onMouseLeave={(e) => {
                                e.stopPropagation();
                                e.target.style.color = '#8e8ea0';
                                e.target.style.backgroundColor = 'transparent';
                              }}
                              title={`Delete ${doc.filename}`}
                            >
                              <Trash2 size={14} />
                            </button>
                          </div>
                        ))}
                      </div>
                    </>
                  )}

                  <button
                    onClick={() => fileInputRef.current?.click()}
                    style={addFilesButtonStyles}
                    onMouseEnter={(e) => Object.assign(e.target.style, addFilesButtonHoverStyles)}
                    onMouseLeave={(e) => e.target.style.backgroundColor = '#19c37d'}
                  >
                    <Paperclip size={16} />
                    Upload Files
                  </button>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".txt,.pdf,.docx,.png,.jpg,.jpeg,.gif,.bmp,.tiff,.webp,.mp3,.wav,.m4a,.ogg"
                    onChange={handleFileSelect}
                    style={{display: 'none'}}
                    multiple
                  />
                </div>
              )}
            </div>

            {/* Send Button */}
            <button
              onClick={handleSendMessage}
              disabled={!inputMessage.trim() || isLoading}
              style={{
                ...sendButtonStyles,
                ...(!inputMessage.trim() || isLoading ? sendButtonDisabledStyles : {})
              }}
              onMouseEnter={(e) => {
                if (inputMessage.trim() && !isLoading) {
                  Object.assign(e.target.style, sendButtonHoverStyles);
                }
              }}
              onMouseLeave={(e) => {
                if (inputMessage.trim() && !isLoading) {
                  e.target.style.backgroundColor = '#19c37d';
                  e.target.style.transform = 'scale(1)';
                }
              }}
              title="Send message"
            >
              <Send size={16} />
            </button>
          </div>
        </div>

        <div style={helperTextStyles}>
          StarRAG Bot - Press Enter to send, Shift+Enter for new line • 
          Supports PDF, DOCX, TXT, images, and audio files (MP3, WAV, M4A, OGG)
        </div>
      </div>
    </div>
  );
};

export default InputArea;