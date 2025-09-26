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
      setTimeout(() => setCopiedIndex(null), 2000); // Reset after 2 seconds
    } catch (err) {
      console.error('Failed to copy text: ', err);
      // Fallback for older browsers
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
    // You could send this feedback to the backend here
    console.log('Message marked as helpful:', index);
  };

  return (
    <div className="flex-1 overflow-y-auto">
      {messages.length === 0 ? (
        <div className="flex flex-col items-center justify-center h-full text-gray-500">
          <div className="w-16 h-16 bg-gradient-to-r from-purple-100 to-blue-100 rounded-full flex items-center justify-center mb-6">
            <Bot className="w-8 h-8 text-purple-600" />
          </div>
          <h2 className="text-2xl font-semibold text-gray-900 mb-4">Welcome to StarRAG Bot</h2>
          <p className="text-center max-w-md text-gray-600 mb-6">
            Upload documents and ask me anything about their content!
          </p>
        </div>
      ) : (
        <>
          {messages.map((message, index) => (
            <div
              key={index}
              className={`group w-full border-b border-gray-200 ${
                message.type === 'user' ? 'bg-white' : 'bg-gray-50'
              }`}
            >
              <div className="m-auto flex gap-6 p-4 text-base md:max-w-2xl lg:max-w-[38rem] xl:max-w-3xl">
                <div className="min-w-[30px]">
                  {message.type === 'user' ? (
                    <User className="w-6 h-6 text-gray-600" />
                  ) : (
                    <Bot className="w-6 h-6 text-purple-600" />
                  )}
                </div>
                <div className="prose prose-gray max-w-none flex-1">
                  <div className="text-gray-900 markdown-content" style={{
                    lineHeight: '1.6'
                  }}>
                    <ReactMarkdown
                      remarkPlugins={[remarkMath]}
                      rehypePlugins={[rehypeKatex]}
                      components={{
                        h1: ({node, ...props}) => <h1 className="text-xl font-bold mb-3 text-gray-900" {...props} />,
                        h2: ({node, ...props}) => <h2 className="text-lg font-semibold mb-2 text-gray-800" {...props} />,
                        h3: ({node, ...props}) => <h3 className="text-md font-medium mb-2 text-gray-700" {...props} />,
                        p: ({node, ...props}) => <p className="mb-2 leading-relaxed" {...props} />,
                        ul: ({node, ...props}) => <ul className="list-disc ml-4 mb-2 space-y-1" {...props} />,
                        ol: ({node, ...props}) => <ol className="list-decimal ml-4 mb-2 space-y-1" {...props} />,
                        li: ({node, ...props}) => <li className="leading-relaxed block" {...props} />,
                        strong: ({node, ...props}) => <strong className="font-semibold text-gray-900" {...props} />,
                        em: ({node, ...props}) => <em className="italic" {...props} />,
                        code: ({node, inline, ...props}) => 
                          inline ? 
                            <code className="bg-gray-100 px-1 py-0.5 rounded text-sm font-mono" {...props} /> :
                            <code className="block bg-gray-100 p-3 rounded-lg text-sm font-mono overflow-x-auto" {...props} />,
                        blockquote: ({node, ...props}) => <blockquote className="border-l-4 border-gray-300 pl-4 italic text-gray-600" {...props} />
                      }}
                    >
                      {message.content}
                    </ReactMarkdown>
                  </div>
                  
                  {message.sources && message.sources.length > 0 && (
                    <div className="mt-4 bg-white border border-gray-200 rounded-lg p-4">
                      <h4 className="text-sm font-semibold text-gray-700 mb-2 flex items-center gap-2">
                        <FileText className="w-4 h-4" />
                        Sources
                      </h4>
                      <div className="space-y-2">
                        {message.sources.map((source, idx) => (
                          <div key={idx} className="flex items-center gap-2 p-2 bg-gray-50 rounded-md">
                            <FileText className="w-4 h-4 text-gray-500" />
                            <span className="text-sm text-gray-600">{source}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                  
                  <div className="flex items-center justify-between mt-4 pt-4 border-t border-gray-200">
                    <div className="flex items-center gap-4">
                      {message.type === 'user' ? (
                        // Copy button for user questions
                        <button 
                          onClick={() => copyToClipboard(message.content, index)}
                          className="flex items-center gap-1 text-gray-500 hover:text-gray-700 transition-colors"
                          title="Copy question"
                        >
                          {copiedIndex === index ? (
                            <>
                              <Check className="w-4 h-4 text-green-600" />
                              <span className="text-sm text-green-600">Copied!</span>
                            </>
                          ) : (
                            <>
                              <Copy className="w-4 h-4" />
                              <span className="text-sm">Copy</span>
                            </>
                          )}
                        </button>
                      ) : (
                        // Both Copy and Helpful buttons for bot answers
                        <>
                          <button 
                            onClick={() => copyToClipboard(message.content, index)}
                            className="flex items-center gap-1 text-gray-500 hover:text-gray-700 transition-colors"
                            title="Copy answer"
                          >
                            {copiedIndex === index ? (
                              <>
                                <Check className="w-4 h-4 text-green-600" />
                                <span className="text-sm text-green-600">Copied!</span>
                              </>
                            ) : (
                              <>
                                <Copy className="w-4 h-4" />
                                <span className="text-sm">Copy</span>
                              </>
                            )}
                          </button>
                          <button 
                            onClick={() => markAsHelpful(index)}
                            className={`flex items-center gap-1 transition-colors ${
                              helpfulIndex === index 
                                ? 'text-green-600' 
                                : 'text-gray-500 hover:text-gray-700'
                            }`}
                            title="Mark as helpful"
                          >
                            <ThumbsUp className="w-4 h-4" />
                            <span className="text-sm">
                              {helpfulIndex === index ? 'Thanks!' : 'Helpful'}
                            </span>
                          </button>
                        </>
                      )}
                    </div>
                    <div className="text-xs text-gray-500">
                      {formatTime(message.timestamp)}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ))}
          
          {isLoading && (
            <div className="group w-full text-gray-800 bg-gray-50 border-b border-gray-200">
              <div className="m-auto flex gap-6 p-4 text-base md:max-w-2xl lg:max-w-[38rem] xl:max-w-3xl">
                <div className="min-w-[30px]">
                  <Bot className="w-6 h-6 text-purple-600" />
                </div>
                <div className="flex items-center gap-2">
                  <Loader2 className="w-4 h-4 animate-spin text-gray-500" />
                  <span className="text-gray-600">StarRAG is thinking...</span>
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