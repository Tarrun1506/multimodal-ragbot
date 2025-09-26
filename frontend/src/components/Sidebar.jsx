import React from 'react';
import { Plus, MessageSquare, History, Loader2 } from 'lucide-react';

const Sidebar = ({
  conversations,
  selectedConversationId,
  isLoadingConversations,
  onNewChat,
  onSelectConversation,
  sidebarOpen
}) => {
  // ChatGPT-like dark theme styles
  const sidebarStyles = {
    width: sidebarOpen ? '260px' : '0',
    backgroundColor: '#202123',
    borderRight: '1px solid #4a4a4a',
    display: 'flex',
    flexDirection: 'column',
    height: '100%',
    transition: 'width 0.3s ease-in-out',
    overflow: 'hidden',
    flexShrink: 0,
  };

  const newChatButtonStyles = {
    width: 'calc(100% - 32px)',
    margin: '16px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'flex-start',
    gap: '12px',
    backgroundColor: 'transparent',
    color: '#ffffff',
    border: '1px solid #565869',
    padding: '12px 16px',
    borderRadius: '6px',
    cursor: 'pointer',
    fontSize: '14px',
    fontWeight: 400,
    transition: 'background-color 0.2s, border-color 0.2s',
  };

  const newChatButtonHoverStyles = {
    backgroundColor: '#343541',
    borderColor: '#8e8ea0',
  };

  const contentAreaStyles = {
    flex: 1,
    overflowY: 'auto',
    padding: '0 8px 16px 8px',
  };

  const historyTitleStyles = {
    padding: '8px 12px 4px 12px',
    fontSize: '12px',
    fontWeight: 500,
    color: '#8e8ea0',
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
    marginBottom: '4px',
  };

  const loadingContainerStyles = {
    display: 'flex',
    justifyContent: 'center',
    padding: '16px 0',
  };

  const emptyStateStyles = {
    textAlign: 'center',
    padding: '16px 0',
    color: '#8e8ea0',
    fontSize: '14px',
    fontStyle: 'italic',
  };

  const conversationListStyles = {
    display: 'flex',
    flexDirection: 'column',
    gap: '2px',
  };

  const conversationButtonStyles = {
    width: '100%',
    textAlign: 'left',
    padding: '10px 12px',
    borderRadius: '4px',
    border: 'none',
    backgroundColor: 'transparent',
    cursor: 'pointer',
    transition: 'background-color 0.2s',
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
    fontSize: '14px',
    color: '#ececf1',
    fontWeight: 400,
  };

  const conversationButtonSelectedStyles = {
    backgroundColor: '#343541',
    color: '#ffffff',
  };

  const conversationButtonHoverStyles = {
    backgroundColor: '#2a2b32',
  };

  return (
    <div style={sidebarStyles}>
      <div style={{padding: '16px'}}>
        <button
          onClick={onNewChat}
          style={newChatButtonStyles}
          onMouseEnter={(e) => Object.assign(e.target.style, newChatButtonHoverStyles)}
          onMouseLeave={(e) => {
            e.target.style.backgroundColor = 'transparent';
            e.target.style.borderColor = '#565869';
          }}
        >
          <Plus size={16} />
          <span>New chat</span>
        </button>
      </div>

      <div style={contentAreaStyles}>
        <h3 style={historyTitleStyles}>
          <History size={12} style={{display: 'inline', marginRight: '8px'}} />
          History
        </h3>
        
        {isLoadingConversations ? (
          <div style={loadingContainerStyles}>
            <Loader2 size={16} className="animate-spin" color="#8e8ea0" />
          </div>
        ) : conversations.length === 0 ? (
          <div style={emptyStateStyles}>
            No conversations yet
          </div>
        ) : (
          <div style={conversationListStyles}>
            {conversations.map(conv => (
              <button
                key={conv.conversation_id}
                onClick={() => onSelectConversation(conv.conversation_id)}
                style={{
                  ...conversationButtonStyles,
                  ...(selectedConversationId === conv.conversation_id ? conversationButtonSelectedStyles : {})
                }}
                onMouseEnter={(e) => {
                  if (selectedConversationId !== conv.conversation_id) {
                    Object.assign(e.target.style, conversationButtonHoverStyles);
                  }
                }}
                onMouseLeave={(e) => {
                  if (selectedConversationId !== conv.conversation_id) {
                    e.target.style.backgroundColor = 'transparent';
                  }
                }}
              >
                <MessageSquare size={16} color="#8e8ea0" />
                <span style={{
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                  flex: 1,
                }}>
                  {conv.first_query || 'New conversation'}
                </span>
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default Sidebar;