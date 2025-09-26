import React, { useState } from 'react';
import { Bot, User, Loader2, Copy, FileText, Check, ThumbsUp } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';

const ChatArea = ({ messages, isLoading, messagesEndRef }) => {
  const formatTime = (date) => date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  const [copiedIndex, setCopiedIndex] = useState(null);
  const [helpfulIndex, setHelpfulIndex] = useState(null);
  
  const copyToClipboard = async (text, index) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedIndex(index);
      setTimeout(() => setCopiedIndex(null), 2000);
    } catch (err) {
      console.error('Failed to copy text: ', err);
      const textArea = document.createElement('textarea');
      textArea.value = text;
      document.body.appendChild(textArea);
      textArea.select();
      document.execCommand('copy');
      document.body.removeChild(textArea);
      setCopiedIndex(index);
      setTimeout(() => setCopiedIndex(null), 2000);
    }
  };
  
  const markAsHelpful = (index) => {
    setHelpfulIndex(index);
    console.log('Message marked as helpful:', index);
  };

  // ChatGPT-like dark theme styles
  const chatAreaStyles = {
    flex: 1,
    overflowY: 'auto',
    backgroundColor: '#343541',
  };

  const emptyStateStyles = {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    height: '100%',
    color: '#d1d5db',
  };

  const emptyStateIconStyles = {
    width: '64px',
    height: '64px',
    background: 'linear-gradient(135deg, #40414f 0%, #565869 100%)',
    borderRadius: '50%',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: '24px',
  };

  const emptyStateTitleStyles = {
    fontSize: '24px',
    fontWeight: 600,
    color: '#ffffff',
    marginBottom: '8px',
  };

  const emptyStateTextStyles = {
    textAlign: 'center',
    maxWidth: '400px',
    color: '#d1d5db',
    marginBottom: '24px',
    lineHeight: '1.5',
  };

  const messageContainerStyles = {
    width: '100%',
  };

  const messageWrapperStyles = {
    margin: '0 auto',
    display: 'flex',
    gap: '24px',
    padding: '24px 16px',
    maxWidth: '768px',
  };

  const avatarStyles = {
    minWidth: '30px',
    display: 'flex',
    alignItems: 'flex-start',
  };

  const userAvatarStyles = {
    width: '28px',
    height: '28px',
    borderRadius: '4px',
    backgroundColor: '#19c37d',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  };

  const botAvatarStyles = {
    width: '28px',
    height: '28px',
    borderRadius: '4px',
    background: 'linear-gradient(135deg, #8b5cf6 0%, #3b82f6 100%)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  };

  const messageContentStyles = {
    flex: 1,
    color: '#d1d5db',
    lineHeight: '1.6',
  };

  const markdownStyles = {
    fontSize: '16px',
    lineHeight: '1.6',
    color: '#d1d5db',
  };

  const sourcesContainerStyles = {
    marginTop: '16px',
    backgroundColor: '#40414f',
    border: '1px solid #565869',
    borderRadius: '8px',
    padding: '16px',
  };

  const sourcesTitleStyles = {
    fontSize: '14px',
    fontWeight: 600,
    color: '#ffffff',
    marginBottom: '8px',
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
  };

  const sourceItemStyles = {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    padding: '8px',
    backgroundColor: '#343541',
    borderRadius: '6px',
    marginBottom: '4px',
    color: '#d1d5db',
  };

  const actionBarStyles = {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginTop: '16px',
    paddingTop: '16px',
    borderTop: '1px solid #565869',
  };

  const actionButtonsStyles = {
    display: 'flex',
    alignItems: 'center',
    gap: '16px',
  };

  const actionButtonStyles = {
    display: 'flex',
    alignItems: 'center',
    gap: '4px',
    backgroundColor: 'transparent',
    border: 'none',
    color: '#8e8ea0',
    cursor: 'pointer',
    fontSize: '14px',
    transition: 'color 0.2s',
  };

  const timestampStyles = {
    fontSize: '12px',
    color: '#8e8ea0',
  };

  const loadingMessageStyles = {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    color: '#8e8ea0',
    fontSize: '14px',
  };

  // Markdown component styles for dark theme
  const markdownComponents = {
    h1: ({node, ...props}) => <h1 style={{fontSize: '20px', fontWeight: 600, color: '#ffffff', marginBottom: '12px'}} {...props} />,
    h2: ({node, ...props}) => <h2 style={{fontSize: '18px', fontWeight: 600, color: '#ffffff', marginBottom: '8px'}} {...props} />,
    h3: ({node, ...props}) => <h3 style={{fontSize: '16px', fontWeight: 600, color: '#ffffff', marginBottom: '8px'}} {...props} />,
    p: ({node, ...props}) => <p style={{marginBottom: '12px', lineHeight: '1.6', color: '#d1d5db'}} {...props} />,
    ul: ({node, ...props}) => <ul style={{listStyleType: 'disc', marginLeft: '20px', marginBottom: '12px', color: '#d1d5db'}} {...props} />,
    ol: ({node, ...props}) => <ol style={{listStyleType: 'decimal', marginLeft: '20px', marginBottom: '12px', color: '#d1d5db'}} {...props} />,
    li: ({node, ...props}) => <li style={{marginBottom: '4px', display: 'list-item', color: '#d1d5db'}} {...props} />,
    strong: ({node, ...props}) => <strong style={{fontWeight: 600, color: '#ffffff'}} {...props} />,
    em: ({node, ...props}) => <em style={{fontStyle: 'italic', color: '#d1d5db'}} {...props} />,
    code: ({node, inline, ...props}) => 
      inline ? 
        <code style={{
          backgroundColor: '#40414f',
          color: '#ffffff',
          padding: '2px 6px',
          borderRadius: '4px',
          fontSize: '14px',
          fontFamily: 'Monaco, Consolas, "Courier New", monospace',
        }} {...props} /> :
        <code style={{
          display: 'block',
          backgroundColor: '#40414f',
          color: '#ffffff',
          padding: '16px',
          borderRadius: '8px',
          fontSize: '14px',
          fontFamily: 'Monaco, Consolas, "Courier New", monospace',
          overflowX: 'auto',
          marginBottom: '12px',
        }} {...props} />,
    blockquote: ({node, ...props}) => <blockquote style={{
      borderLeft: '4px solid #565869',
      paddingLeft: '16px',
      fontStyle: 'italic',
      color: '#8e8ea0',
      marginBottom: '12px',
    }} {...props} />
  };

  return (
    <div style={chatAreaStyles}>
      {messages.length === 0 ? (
        <div style={emptyStateStyles}>
          <div style={emptyStateIconStyles}>
            <Bot size={32} color="#d1d5db" />
          </div>
          <h2 style={emptyStateTitleStyles}>Welcome to StarRAG Bot</h2>
          <p style={emptyStateTextStyles}>
            Upload documents and ask me anything about their content!
          </p>
        </div>
      ) : (
        <>
          {messages.map((message, index) => (
            <div
              key={index}
              style={{
                ...messageContainerStyles,
                backgroundColor: message.type === 'user' ? '#343541' : '#444654',
              }}
            >
              <div style={messageWrapperStyles}>
                <div style={avatarStyles}>
                  {message.type === 'user' ? (
                    <div style={userAvatarStyles}>
                      <User size={16} color="#ffffff" />
                    </div>
                  ) : (
                    <div style={botAvatarStyles}>
                      <Bot size={16} color="#ffffff" />
                    </div>
                  )}
                </div>
                <div style={messageContentStyles}>
                  <div style={markdownStyles}>
                    <ReactMarkdown
                      remarkPlugins={[remarkMath]}
                      rehypePlugins={[rehypeKatex]}
                      components={markdownComponents}
                    >
                      {message.content}
                    </ReactMarkdown>
                  </div>
                  
                  {message.sources && message.sources.length > 0 && (
                    <div style={sourcesContainerStyles}>
                      <h4 style={sourcesTitleStyles}>
                        <FileText size={16} />
                        Sources
                      </h4>
                      <div style={{display: 'flex', flexDirection: 'column', gap: '8px'}}>
                        {message.sources.map((source, idx) => (
                          <div key={idx} style={sourceItemStyles}>
                            <FileText size={16} color="#8e8ea0" />
                            <span style={{fontSize: '14px'}}>{source}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                  
                  <div style={actionBarStyles}>
                    <div style={actionButtonsStyles}>
                      {message.type === 'user' ? (
                        <button 
                          onClick={() => copyToClipboard(message.content, index)}
                          style={actionButtonStyles}
                          onMouseEnter={(e) => e.target.style.color = '#ffffff'}
                          onMouseLeave={(e) => e.target.style.color = '#8e8ea0'}
                          title="Copy question"
                        >
                          {copiedIndex === index ? (
                            <>
                              <Check size={16} color="#19c37d" />
                              <span style={{color: '#19c37d'}}>Copied!</span>
                            </>
                          ) : (
                            <>
                              <Copy size={16} />
                              <span>Copy</span>
                            </>
                          )}
                        </button>
                      ) : (
                        <>
                          <button 
                            onClick={() => copyToClipboard(message.content, index)}
                            style={actionButtonStyles}
                            onMouseEnter={(e) => e.target.style.color = '#ffffff'}
                            onMouseLeave={(e) => e.target.style.color = '#8e8ea0'}
                            title="Copy answer"
                          >
                            {copiedIndex === index ? (
                              <>
                                <Check size={16} color="#19c37d" />
                                <span style={{color: '#19c37d'}}>Copied!</span>
                              </>
                            ) : (
                              <>
                                <Copy size={16} />
                                <span>Copy</span>
                              </>
                            )}
                          </button>
                          <button 
                            onClick={() => markAsHelpful(index)}
                            style={{
                              ...actionButtonStyles,
                              color: helpfulIndex === index ? '#19c37d' : '#8e8ea0'
                            }}
                            onMouseEnter={(e) => {
                              if (helpfulIndex !== index) e.target.style.color = '#ffffff'
                            }}
                            onMouseLeave={(e) => {
                              if (helpfulIndex !== index) e.target.style.color = '#8e8ea0'
                            }}
                            title="Mark as helpful"
                          >
                            <ThumbsUp size={16} />
                            <span>{helpfulIndex === index ? 'Thanks!' : 'Helpful'}</span>
                          </button>
                        </>
                      )}
                    </div>
                    <div style={timestampStyles}>
                      {formatTime(message.timestamp)}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ))}
          
          {isLoading && (
            <div style={{...messageContainerStyles, backgroundColor: '#444654'}}>
              <div style={messageWrapperStyles}>
                <div style={avatarStyles}>
                  <div style={botAvatarStyles}>
                    <Bot size={16} color="#ffffff" />
                  </div>
                </div>
                <div style={loadingMessageStyles}>
                  <Loader2 size={16} className="animate-spin" />
                  <span>StarRAG is thinking...</span>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </>
      )}
    </div>
  );
};

export default ChatArea;