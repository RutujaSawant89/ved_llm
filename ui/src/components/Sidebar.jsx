import { NavLink } from 'react-router-dom'
import { MessageSquare, Folder } from 'lucide-react'

const links = [
  { to: '/', icon: MessageSquare, label: 'Chat' },
  { to: '/documents', icon: Folder, label: 'Documents' },
]

export default function Sidebar() {
  return (
    <nav className="sidebar">
      <span className="sidebar-logo">T</span>
      <div className="sidebar-nav">
        {links.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) => `sidebar-link${isActive ? ' active' : ''}`}
            title={label}
          >
            <Icon size={18} strokeWidth={1.5} />
          </NavLink>
        ))}
      </div>
    </nav>
  )
}
