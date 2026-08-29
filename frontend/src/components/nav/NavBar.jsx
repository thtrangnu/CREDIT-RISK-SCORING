import { NavLink } from 'react-router-dom';
import { useEffect, useState } from 'react';
import './nav-bar.css';

const LINKS = [
  { to: '/', label: 'Score', end: true },
  { to: '/insights', label: 'Insights' },
  { to: '/history', label: 'History' },
];

function useTheme() {
  const [theme, setTheme] = useState(() => localStorage.getItem('theme') ?? 'dark');

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
  }, [theme]);

  return [theme, setTheme];
}

export default function NavBar() {
  const [theme, setTheme] = useTheme();

  return (
    <header className="nav-bar">
      <div className="container nav-bar__inner">
        <div className="nav-bar__brand">
          <span className="nav-bar__mark" aria-hidden="true">
            ◆
          </span>
          <div>
            <div className="nav-bar__title">Home Credit Risk Console</div>
            <div className="nav-bar__subtitle eyebrow">Default risk scoring</div>
          </div>
        </div>

        <nav aria-label="Điều hướng chính" className="nav-bar__links">
          {LINKS.map((link) => (
            <NavLink
              key={link.to}
              to={link.to}
              end={link.end}
              className={({ isActive }) => `nav-bar__link${isActive ? ' is-active' : ''}`}
            >
              {link.label}
            </NavLink>
          ))}
        </nav>

        <button
          type="button"
          className="nav-bar__theme-toggle"
          onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
          aria-label={`Chuyển sang giao diện ${theme === 'dark' ? 'sáng' : 'tối'}`}
        >
          {theme === 'dark' ? '☾' : '☀'}
        </button>
      </div>
    </header>
  );
}
