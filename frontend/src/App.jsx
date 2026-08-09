import { useState, useEffect } from 'react'

export default function App() {
  const [activeTab, setActiveTab] = useState('search') // 'search' | 'compare' | 'news' | 'admin'
  const [role, setRole] = useState('user') // 'user' | 'admin'

  // Search State
  const [question, setQuestion] = useState('What are the maximum permitted limits for preservatives in fruit squashes?')
  const [category, setCategory] = useState('All')
  const [searchResult, setSearchResult] = useState(null)
  const [searching, setSearching] = useState(false)

  // Compare State
  const [compareParam, setCompareParam] = useState('Monosodium Glutamate (MSG)')
  const [compareResult, setCompareResult] = useState(null)
  const [comparing, setComparing] = useState(false)

  // News State
  const [newsList, setNewsList] = useState([])
  const [loadingNews, setLoadingNews] = useState(false)

  // Admin State
  const [adminMetrics, setAdminMetrics] = useState(null)
  const [loadingAdmin, setLoadingAdmin] = useState(false)

  // Execute FSSAI Search
  async function handleSearch(e) {
    if (e) e.preventDefault()
    setSearching(true)
    try {
      const res = await fetch('/api/search', {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': role === 'admin' ? 'Admin token' : 'User token'
        },
        body: JSON.stringify({ question, category })
      })
      const data = await res.json()
      setSearchResult(data)
    } catch (err) {
      console.error(err)
    } finally {
      setSearching(false)
    }
  }

  // Execute Global Comparison
  async function handleCompare(e) {
    if (e) e.preventDefault()
    setComparing(true)
    try {
      const res = await fetch('/api/compare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ parameter: compareParam })
      })
      const data = await res.json()
      setCompareResult(data)
    } catch (err) {
      console.error(err)
    } finally {
      setComparing(false)
    }
  }

  // Fetch News Feed
  async function fetchNews() {
    setLoadingNews(true)
    try {
      const res = await fetch('/api/news')
      const data = await res.json()
      setNewsList(data)
    } catch (err) {
      console.error(err)
    } finally {
      setLoadingNews(false)
    }
  }

  // Fetch Admin Metrics
  async function fetchAdminMetrics() {
    setLoadingAdmin(true)
    try {
      const res = await fetch('/api/admin/metrics', {
        headers: { 'Authorization': 'Admin token' }
      })
      const data = await res.json()
      setAdminMetrics(data)
    } catch (err) {
      console.error(err)
    } finally {
      setLoadingAdmin(false)
    }
  }

  useEffect(() => {
    if (activeTab === 'news') fetchNews()
    if (activeTab === 'admin') fetchAdminMetrics()
  }, [activeTab])

  return (
    <div className="layout">
      {/* Top Navbar */}
      <header className="navbar">
        <div className="brand">
          <span className="brand-logo">🛡️</span>
          <div>
            <h1 className="brand-title">NiyamAI</h1>
            <p className="brand-sub">FSSAI Food Safety Regulatory Intelligence</p>
          </div>
        </div>

        <nav className="nav-tabs">
          <button 
            className={`nav-btn ${activeTab === 'search' ? 'active' : ''}`}
            onClick={() => setActiveTab('search')}
          >
            🔍 FSSAI Search & Citations
          </button>
          <button 
            className={`nav-btn ${activeTab === 'compare' ? 'active' : ''}`}
            onClick={() => setActiveTab('compare')}
          >
            🌐 Global Comparison
          </button>
          <button 
            className={`nav-btn ${activeTab === 'news' ? 'active' : ''}`}
            onClick={() => setActiveTab('news')}
          >
            📰 Updates & Fines Feed
          </button>
          <button 
            className={`nav-btn ${activeTab === 'admin' ? 'active' : ''}`}
            onClick={() => setActiveTab('admin')}
          >
            📊 Admin Telemetry
          </button>
        </nav>

        <div className="user-role-toggle">
          <span className="role-label">Role:</span>
          <select 
            value={role} 
            onChange={(e) => setRole(e.target.value)}
            className="role-select"
          >
            <option value="user">User (FSSAI Searcher)</option>
            <option value="admin">Admin (Metrics & Costs)</option>
          </select>
        </div>
      </header>

      {/* Main Workspace Content */}
      <main className="content-area">

        {/* TAB 1: FSSAI Search Workspace */}
        {activeTab === 'search' && (
          <div className="grid-workspace">
            <section className="search-pane">
              <h2>🔍 Search FSSAI Regulations</h2>
              <form onSubmit={handleSearch} className="search-form">
                <div className="form-group">
                  <label htmlFor="category">Category Filter</label>
                  <select 
                    id="category"
                    value={category} 
                    onChange={(e) => setCategory(e.target.value)}
                    className="input-field"
                  >
                    <option value="All">All Regulations</option>
                    <option value="Additives">Food Additives & Preservatives</option>
                    <option value="Packaging">Packaging & Labeling (2020)</option>
                    <option value="Licensing">Licensing & Registration</option>
                  </select>
                </div>

                <div className="form-group">
                  <label htmlFor="question">Regulatory Query / Requirement</label>
                  <textarea 
                    id="question"
                    value={question} 
                    onChange={(e) => setQuestion(e.target.value)}
                    rows={4}
                    className="input-field"
                  />
                </div>

                <button type="submit" className="primary-btn" disabled={searching}>
                  {searching ? 'Analyzing Regulations...' : 'Run FSSAI Query'}
                </button>
              </form>

              {searchResult && (
                <div className="telemetry-bar">
                  <span>⏱️ Latency: <strong>{searchResult.execution_time_ms} ms</strong></span>
                  <span>💰 Est. Cost: <strong>${searchResult.estimated_cost_usd}</strong></span>
                  <span>🎯 Confidence: <strong className="badge-high">{searchResult.confidence_label} ({Math.round(searchResult.confidence_score * 100)}%)</strong></span>
                </div>
              )}
            </section>

            {/* Answer & Citations Pane */}
            <section className="results-pane">
              <h2>📖 Explainable Analysis & Verbatim Citations</h2>
              {!searchResult && (
                <div className="empty-state">
                  <p>Enter an FSSAI regulatory question on the left to view step-by-step analysis and verified citations.</p>
                </div>
              )}

              {searchResult && (
                <div className="results-card">
                  <div className="verdict-box">
                    <h3>Executive Summary</h3>
                    <p>{searchResult.summary}</p>
                  </div>

                  <div className="analysis-box">
                    <h3>Regulatory Analysis</h3>
                    <p>{searchResult.detailed_analysis}</p>
                  </div>

                  <div className="citations-box">
                    <h3>Verbatim Legal Citations</h3>
                    {searchResult.citations.map((cite, idx) => (
                      <div key={idx} className="citation-card">
                        <div className="citation-header">
                          <span className="doc-name">📜 {cite.document_name}</span>
                          <span className="sec-tag">{cite.section} (Page {cite.page_number})</span>
                        </div>
                        <blockquote className="quote">"{cite.verbatim_text}"</blockquote>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </section>
          </div>
        )}

        {/* TAB 2: Global Comparison Matrix */}
        {activeTab === 'compare' && (
          <div className="single-pane">
            <h2>🌐 Global Food Safety Standard Benchmark</h2>
            <p className="pane-desc">Compare FSSAI (India) rules against US FDA and EU EFSA regulations side-by-side.</p>

            <form onSubmit={handleCompare} className="compare-form">
              <input 
                type="text" 
                value={compareParam}
                onChange={(e) => setCompareParam(e.target.value)}
                placeholder="e.g. Monosodium Glutamate, Trans Fats, Tartrazine"
                className="input-field compare-input"
              />
              <button type="submit" className="primary-btn" disabled={comparing}>
                {comparing ? 'Comparing...' : 'Compare Regions'}
              </button>
            </form>

            {compareResult && (
              <div className="compare-results">
                <div className="takeaway-banner">
                  💡 <strong>Key Takeaway:</strong> {compareResult.key_takeaway}
                </div>

                <div className="matrix-table-container">
                  <table className="matrix-table">
                    <thead>
                      <tr>
                        <th>Region & Regulator</th>
                        <th>Status</th>
                        <th>Permitted Limit / Rule</th>
                        <th>Citation / Reference</th>
                      </tr>
                    </thead>
                    <tbody>
                      {compareResult.standards.map((std, idx) => (
                        <tr key={idx}>
                          <td><strong>{std.region}</strong> ({std.authority})</td>
                          <td>
                            <span className={`status-pill status-${std.status.toLowerCase().replace(/\s+/g, '-')}`}>
                              {std.status}
                            </span>
                          </td>
                          <td>{std.permitted_limit}</td>
                          <td className="code-cite">{std.citation}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        )}

        {/* TAB 3: Updates & Fines Feed */}
        {activeTab === 'news' && (
          <div className="single-pane">
            <h2>📰 FSSAI Updates, Advisories & Enforcement Notices</h2>
            <p className="pane-desc">Latest advisory circulars, penalty notices, and regulatory amendments.</p>

            {loadingNews ? <p>Loading updates...</p> : (
              <div className="news-list">
                {newsList.map((item) => (
                  <article key={item.id} className="news-card">
                    <div className="news-meta">
                      <span className="news-cat">{item.category}</span>
                      <span className="news-date">{item.date}</span>
                    </div>
                    <h3>{item.title}</h3>
                    <p>{item.summary}</p>
                    <a href={item.source_url} target="_blank" rel="noreferrer" className="news-link">
                      View Official Source ↗
                    </a>
                  </article>
                ))}
              </div>
            )}
          </div>
        )}

        {/* TAB 4: Admin Telemetry Dashboard */}
        {activeTab === 'admin' && (
          <div className="single-pane">
            <h2>📊 Admin Telemetry & API Cost Dashboard</h2>
            <p className="pane-desc">Monitor total requests, average latency, token counts, and estimated LLM cost per request.</p>

            {loadingAdmin ? <p>Loading telemetry logs...</p> : adminMetrics && (
              <div>
                <div className="stats-cards-grid">
                  <div className="stat-card">
                    <span className="stat-label">Total Requests</span>
                    <span className="stat-val">{adminMetrics.total_requests}</span>
                  </div>
                  <div className="stat-card">
                    <span className="stat-label">Total API Cost</span>
                    <span className="stat-val">${adminMetrics.total_cost_usd}</span>
                  </div>
                  <div className="stat-card">
                    <span className="stat-label">Avg Response Latency</span>
                    <span className="stat-val">{adminMetrics.avg_response_time_ms} ms</span>
                  </div>
                  <div className="stat-card">
                    <span className="stat-label">Active Users</span>
                    <span className="stat-val">{adminMetrics.active_users_count}</span>
                  </div>
                </div>

                <h3 className="section-subtitle">Per-Request Telemetry & Cost Logs</h3>
                <div className="matrix-table-container">
                  <table className="matrix-table">
                    <thead>
                      <tr>
                        <th>User ID</th>
                        <th>Endpoint</th>
                        <th>Latency</th>
                        <th>Prompt Tokens</th>
                        <th>Completion Tokens</th>
                        <th>Est. Cost ($)</th>
                        <th>Timestamp</th>
                      </tr>
                    </thead>
                    <tbody>
                      {adminMetrics.logs.length === 0 ? (
                        <tr>
                          <td colSpan={7} style={{ textAlign: 'center' }}>No requests recorded yet. Run a search query!</td>
                        </tr>
                      ) : (
                        adminMetrics.logs.map((log, idx) => (
                          <tr key={idx}>
                            <td><code>{log.user_id}</code></td>
                            <td>{log.endpoint}</td>
                            <td>{log.execution_time_ms} ms</td>
                            <td>{log.prompt_tokens}</td>
                            <td>{log.completion_tokens}</td>
                            <td><strong>${log.estimated_cost_usd}</strong></td>
                            <td className="code-cite">{log.timestamp.split('T')[1].slice(0, 8)}</td>
                          </tr>
                        ))
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  )
}
