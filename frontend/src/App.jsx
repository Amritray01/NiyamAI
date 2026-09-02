import { useState, useEffect } from 'react'

const SUGGESTED_QUERIES = [
  {
    category: "Contaminants & Heavy Metals",
    filter: "All",
    icon: "🧪",
    queries: [
      "What are the maximum permitted limits for lead in various food products ?",
      "What is the maximum permitted lead limit in infant milk food?",
      "What is the maximum limit of arsenic permitted in fruit juices and nectars?",
      "What is the maximum permitted level of Aflatoxin B1 in food items?"
    ]
  },
  {
    category: "Honey & Sweetening Agents",
    filter: "All",
    icon: "🍯",
    queries: [
      "What is the maximum permissible limit of moisture in natural honey?",
      "What is the maximum sucrose content allowed in blossom honey?",
      "What exception applies to the stated moisture limit in Caravia / Trigona bee honey?"
    ]
  },
  {
    category: "Prohibitions & Adulteration",
    filter: "All",
    icon: "🚫",
    queries: [
      "What is the restriction under Section 2.1 of Prohibition Regulations for selling milk containing added water?",
      "What restrictions apply to the sale of ghee containing added vegetable fats?"
    ]
  },
  {
    category: "Advertising & Claims",
    filter: "Packaging",
    icon: "🏷️",
    queries: [
      "What conditions govern non-addition of sugar claims on food packaging?",
      "What general hygiene and safety principles apply to advertising food products to children?"
    ]
  }
]

export default function App() {
  const [activeTab, setActiveTab] = useState('search') // 'search' | 'news'

  // Search Workspace State
  const [question, setQuestion] = useState('What are the maximum permitted limits for lead in various food products ?')
  const [category, setCategory] = useState('All')
  const [searchResult, setSearchResult] = useState(null)
  const [searching, setSearching] = useState(false)
  const [searchError, setSearchError] = useState(null)

  // News State
  const [newsData, setNewsData] = useState({
    ai_overview: '',
    category_summaries: {},
    news_items: []
  })
  const [newsFilter, setNewsFilter] = useState('All')
  const [newsSearch, setNewsSearch] = useState('')
  const [loadingNews, setLoadingNews] = useState(false)

  // Execute FSSAI Search Query
  async function handleSearch(e, customQuery = null, customCategory = null) {
    if (e) e.preventDefault()
    const queryText = customQuery !== null ? customQuery : question
    const queryCategory = customCategory !== null ? customCategory : category
    if (!queryText.trim()) return

    setSearching(true)
    setSearchError(null)

    try {
      const res = await fetch('/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: queryText, category: queryCategory })
      })

      if (!res.ok) {
        throw new Error(`Server returned status ${res.status}`)
      }

      const data = await res.json()
      setSearchResult(data)
    } catch (err) {
      console.error("Search Error:", err)
      setSearchError("Failed to fetch response from FastAPI backend. Ensure backend server is running on port 8000.")
    } finally {
      setSearching(false)
    }
  }

  // Quick prompt select & execute
  function applySuggestion(qText, filterCat = "All") {
    setQuestion(qText)
    setCategory(filterCat)
    handleSearch(null, qText, filterCat)
  }

  // Fetch Live News Feed
  async function fetchNews() {
    setLoadingNews(true)
    try {
      const res = await fetch('/api/news')
      const data = await res.json()
      if (data && data.news_items) {
        setNewsData(data)
      } else if (Array.isArray(data)) {
        setNewsData({ ai_overview: '', category_summaries: {}, news_items: data })
      }
    } catch (err) {
      console.error("News Fetch Error:", err)
    } finally {
      setLoadingNews(false)
    }
  }

  // Handle Tab Switch Actions
  useEffect(() => {
    if (activeTab === 'news') fetchNews()
  }, [activeTab])

  // Filtered news items
  const filteredNews = (newsData.news_items || []).filter(item => {
    const matchesCat = newsFilter === 'All' || item.category === newsFilter
    const matchesSearch = !newsSearch.trim() || 
      item.title.toLowerCase().includes(newsSearch.toLowerCase()) ||
      (item.summary && item.summary.toLowerCase().includes(newsSearch.toLowerCase()))
    return matchesCat && matchesSearch
  })

  // Confidence badge color helper
  function getConfidenceBadgeClass(label, score) {
    if (label === 'High' || score >= 0.75) return 'badge-high'
    if (label === 'Medium' || score >= 0.50) return 'badge-medium'
    return 'badge-low'
  }

  return (
    <div className="layout">
      {/* Top Navbar Header */}
      <header className="navbar">
        <div className="navbar-left">
          <div className="brand">
            <span className="brand-logo">🛡️</span>
            <div>
              <h1 className="brand-title">NiyamAI</h1>
              <p className="brand-sub">FSSAI Regulatory & Compliance Search</p>
            </div>
          </div>

          {/* Top-Left Main Menu Tabs */}
          <nav className="nav-tabs-left">
            <button 
              className={`nav-btn ${activeTab === 'search' ? 'active' : ''}`}
              onClick={() => setActiveTab('search')}
            >
              🔍 FSSAI RAG Search
            </button>
            <button 
              className={`nav-btn ${activeTab === 'news' ? 'active' : ''}`}
              onClick={() => setActiveTab('news')}
            >
              📰 Live Intelligence & News
            </button>
          </nav>
        </div>
      </header>

      {/* Main Workspace */}
      <main className="content-area">

        {/* TAB 1: FSSAI RAG Search Workspace */}
        {activeTab === 'search' && (
          <div>
            {/* Quick Test Queries & Golden Dataset Suggestions */}
            <div className="suggestions-panel">
              <div className="suggestions-header">
                <span className="suggestions-title">💡 Quick Test Queries from Evaluation Golden Dataset:</span>
                <span className="suggestions-hint">Click any query to run instant RAG analysis</span>
              </div>
              <div className="suggestions-grid">
                {SUGGESTED_QUERIES.map((group, gIdx) => (
                  <div key={gIdx} className="suggestion-group">
                    <span className="group-label">{group.icon} {group.category}</span>
                    <div className="chip-list">
                      {group.queries.map((q, qIdx) => (
                        <button
                          key={qIdx}
                          type="button"
                          className="suggestion-chip"
                          onClick={() => applySuggestion(q, group.filter)}
                        >
                          {q}
                        </button>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="grid-workspace">
              <section className="search-pane">
                <h2>🔍 FSSAI Regulation Query Workspace</h2>
                <p className="pane-desc">Ask any food safety, heavy metal limit, contaminant, labeling, or licensing compliance question.</p>

                <form onSubmit={handleSearch} className="search-form">
                  <div className="form-group">
                    <label htmlFor="category">Category Filter</label>
                    <select 
                      id="category"
                      value={category} 
                      onChange={(e) => setCategory(e.target.value)}
                      className="input-field"
                    >
                      <option value="All">All FSSAI Regulations</option>
                      <option value="Contaminants">Contaminants & Heavy Metals (2011)</option>
                      <option value="Additives">Food Additives & Preservatives (2011)</option>
                      <option value="Packaging">Packaging & Labeling (2020)</option>
                      <option value="Licensing">Licensing & Registration (2011)</option>
                      <option value="Microbiological">Microbiological Requirements</option>
                    </select>
                  </div>

                  <div className="form-group">
                    <label htmlFor="question">Compliance Question / Query</label>
                    <textarea 
                      id="question"
                      value={question} 
                      onChange={(e) => setQuestion(e.target.value)}
                      rows={3}
                      placeholder="Enter your regulatory query..."
                      className="input-field"
                    />
                  </div>

                  <button type="submit" className="primary-btn" disabled={searching}>
                    {searching ? 'Analyzing Regulations...' : 'Run FSSAI Query 🚀'}
                  </button>
                </form>

                {searchError && (
                  <div className="error-banner">
                    ⚠️ {searchError}
                  </div>
                )}

                {searchResult && (
                  <div className="telemetry-bar">
                    <span>⏱️ Latency: <strong>{searchResult.execution_time_ms} ms</strong></span>
                    <span>🎯 Confidence: <strong className={getConfidenceBadgeClass(searchResult.confidence_label, searchResult.confidence_score)}>
                      {searchResult.confidence_label} ({Math.round(searchResult.confidence_score * 100)}%)
                    </strong></span>
                  </div>
                )}
              </section>

              {/* Right Pane: Explainable Results & Citations */}
              <section className="results-pane">
                <h2>📖 Verbatim Legal Citations & Analysis</h2>
                {!searchResult && !searching && (
                  <div className="empty-state">
                    <p>💡 Enter a regulatory question on the left or choose a <strong>suggested test query</strong> above to retrieve exact legal clause citations and step-by-step XAI breakdown.</p>
                  </div>
                )}

                {searching && (
                  <div className="loading-state">
                    <p>⏳ Searching vector store and running RAG reasoning chain...</p>
                  </div>
                )}

                {searchResult && (
                  <div className="results-card">
                    <div className="verdict-box">
                      <h3>Executive Summary</h3>
                      <p>{searchResult.summary}</p>
                    </div>

                    <div className="analysis-box">
                      <h3>Explainable RAG Analysis</h3>
                      <p className="analysis-text">{searchResult.detailed_analysis}</p>
                    </div>

                    <div className="citations-box">
                      <h3>Official Verbatim Legal Citations ({searchResult.citations ? searchResult.citations.length : 0})</h3>
                      {searchResult.citations && searchResult.citations.length > 0 ? (
                        searchResult.citations.map((cite, idx) => (
                          <div key={idx} className="citation-card">
                            <div className="citation-header">
                              <span className="doc-name">📜 {cite.document_name}</span>
                              <span className="sec-tag">{cite.section} (Page {cite.page_number})</span>
                            </div>
                            <blockquote className="quote">"{cite.verbatim_text}"</blockquote>
                          </div>
                        ))
                      ) : (
                        <p className="text-muted">No direct citations returned.</p>
                      )}
                    </div>
                  </div>
                )}
              </section>
            </div>
          </div>
        )}

        {/* TAB 2: Live Updates & Regulatory Intelligence News */}
        {activeTab === 'news' && (
          <div className="single-pane">
            <div className="pane-header-row">
              <div>
                <h2>📰 FSSAI Live Regulatory Intelligence & News Agent</h2>
                <p className="pane-desc">Live updates, official circulars, gazette amendments, and FDA enforcement raids tracked by AI.</p>
              </div>
              <button onClick={fetchNews} className="secondary-btn" disabled={loadingNews}>
                {loadingNews ? 'Refreshing...' : '🔄 Run Live Scrape & Search'}
              </button>
            </div>

            {/* AI Regulatory Pulse Banner */}
            {newsData.ai_overview && (
              <div className="news-ai-banner">
                <div className="news-ai-header">
                  <span>⚡ AI Regulatory Executive Pulse</span>
                </div>
                <p className="news-ai-text">{newsData.ai_overview}</p>
              </div>
            )}

            {/* Category Breakdown Cards */}
            {newsData.category_summaries && Object.keys(newsData.category_summaries).length > 0 && (
              <div className="category-summaries-grid">
                {Object.entries(newsData.category_summaries).map(([cat, summary], idx) => (
                  <div key={idx} className="cat-summary-card">
                    <h4>{cat}</h4>
                    <p>{summary}</p>
                  </div>
                ))}
              </div>
            )}

            {/* Filter & Search Bar */}
            <div className="news-controls-bar">
              <div className="news-filter-pills">
                {['All', 'Advisories', 'Notifications', 'In The News', 'Enforcement & Fine'].map((cat) => (
                  <button
                    key={cat}
                    type="button"
                    className={`filter-pill ${newsFilter === cat ? 'active' : ''}`}
                    onClick={() => setNewsFilter(cat)}
                  >
                    {cat}
                  </button>
                ))}
              </div>
              <input
                type="text"
                placeholder="Filter news by keyword..."
                value={newsSearch}
                onChange={(e) => setNewsSearch(e.target.value)}
                className="input-field news-search-input"
              />
            </div>

            {loadingNews ? (
              <div className="loading-state"><p>⏳ Running Live FSSAI News Agent across portals & enforcement stream...</p></div>
            ) : (
              <div className="news-grid">
                {filteredNews.length > 0 ? (
                  filteredNews.map((item, idx) => (
                    <article key={item.id || idx} className="news-card">
                      <div className="news-meta">
                        <span className="news-cat">{item.category || 'Advisories'}</span>
                        <span className={`severity-tag severity-${(item.severity || 'Medium').toLowerCase()}`}>
                          {item.severity || 'Medium'} Urgency
                        </span>
                        <span className="news-date">{item.date || 'Recent'}</span>
                      </div>
                      <h3>{item.title}</h3>
                      <p className="news-summary">{item.summary}</p>

                      <a href={item.source_url} target="_blank" rel="noreferrer" className="news-link">
                        View Official Regulatory Document ↗
                      </a>
                    </article>
                  ))
                ) : (
                  <p className="text-muted" style={{ padding: '2rem 0' }}>No news items match the current filter.</p>
                )}
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  )
}
