import React from 'react';
import { Menu, Bot } from 'lucide-react';

const Header = ({ 
  onToggleSidebar,
  selectedModel
}) => {
  // ChatGPT-like dark theme styles
  const headerStyles = {
    position: 'sticky',
    top: 0,
    zIndex: 40,
    width: '100%',
    borderBottom: '1px solid #565869',
    backgroundColor: '#343541',
    backdropFilter: 'blur(8px)',
  };

  const containerStyles = {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    height: '56px',
    padding: '0 16px',
  };

  const leftSectionStyles = {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
  };

  const menuButtonStyles = {
    padding: '8px',
    borderRadius: '8px',
    border: 'none',
    backgroundColor: 'transparent',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    transition: 'background-color 0.2s',
    color: '#d1d5db',
  };

  const menuButtonHoverStyles = {
    backgroundColor: '#40414f',
  };

  const logoStyles = {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
  };

  const logoIconStyles = {
    width: '32px',
    height: '32px',
    background: 'linear-gradient(135deg, #8b5cf6 0%, #3b82f6 100%)',
    borderRadius: '8px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  };

  const logoTextStyles = {
    fontSize: '18px',
    fontWeight: 600,
    color: '#ffffff',
  };

  const rightSectionStyles = {
    display: 'flex',
    alignItems: 'center',
    gap: '16px',
  };

  const modelBadgeStyles = {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    padding: '4px 12px',
    fontSize: '14px',
    backgroundColor: '#40414f',
    borderRadius: '20px',
    fontWeight: 500,
    color: '#d1d5db',
  };

  return (
    <header style={headerStyles}>
      <div style={containerStyles}>
        <div style={leftSectionStyles}>
          <button 
            onClick={onToggleSidebar}
            style={menuButtonStyles}
            onMouseEnter={(e) => Object.assign(e.target.style, menuButtonHoverStyles)}
            onMouseLeave={(e) => e.target.style.backgroundColor = 'transparent'}
          >
            <Menu size={20} color="#d1d5db" />
          </button>
          <div style={logoStyles}>
            <div style={logoIconStyles}>
              <Bot size={20} color="#ffffff" />
            </div>
            <span style={logoTextStyles}>StarRAG Bot</span>
          </div>
        </div>

        <div style={rightSectionStyles}>
          <div style={modelBadgeStyles}>
            <Bot size={16} color="#d1d5db" />
            <span>{selectedModel}</span>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;