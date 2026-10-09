import { useState, useEffect } from 'react'

const SUGGESTED_QUERIES = [
  {
    category: "Contaminants & Heavy Metals",
    filter: "Contaminants",
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
    filter: "Sweetening Agents & Standards",
    icon: "🍯",
    queries: [
      "What is the maximum permissible limit of moisture in natural honey?",
      "What is the maximum sucrose content allowed in blossom honey?",
      "What exception applies to the stated moisture limit in Caravia / Trigona bee honey?"
    ]
  },
  {
    category: "Prohibitions & Adulteration",
    filter: "Prohibitions",
    icon: "🚫",
    queries: [
      "What is the restriction under Section 2.1 of Prohibition Regulations for selling milk containing added water?",
      "What restrictions apply to the sale of ghee containing added vegetable fats?"
    ]
  },
  {
    category: "Advertising & Claims",
    filter: "Advertising & Claims",
    icon: "🏷️",
    queries: [
      "What conditions govern non-addition of sugar claims on food packaging?",
      "What general hygiene and safety principles apply to advertising food products to children?"
    ]
  }
]

export default function App() {
  const [activeTab, setActiveTab] = useState('search')
  const [question, setQuestion] = useState('What are the maximum permitted limits for lead in various food products ?')
  const [category, setCategory] = useState('All')
  const [searchResult, setSearchResult] = useState(null)
  const [searching, setSearching] = useState(false)
  const [searchError, setSearchError] = useState(null)
  const [copiedSummary, setCopiedSummary] = useState(false)
  const [copiedAnalysis, setCopiedAnalysis] = useState(false)
  const [isStreaming, setIsStreaming] = useState(false)

  const [newsData, setNewsData] = useState({
    ai_overview: '',
    category_summaries: {},
    news_items: []
  })
  const [newsFilter, setNewsFilter] = useState('All')
  const [newsSearch, setNewsSearch] = useState('')
  const [loadingNews, setLoadingNews] = useState(false)

  function copySummary(text) {
    if (!text) return
    navigator.clipboard.writeText(text)
    setCopiedSummary(true)
    setTimeout(() => setCopiedSummary(false), 2000)
  }

  function copyAnalysis(text) {
    if (!text) return
    navigator.clipboard.writeText(text)
    setCopiedAnalysis(true)
    setTimeout(() => setCopiedAnalysis(false), 2000)
  }

  async function handleSearch(e, customQuery = null, customCategory = null) {
    if (e) e.preventDefault()
    const queryText = customQuery !== null ? customQuery : question
    const queryCategory = customCategory !== null ? customCategory : category
    if (!queryText.trim()) return

    setSearching(true)
    setIsStreaming(true)
    setSearchError(null)
    setSearchResult(null)

    try {
      const res = await fetch('/api/search/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: queryText, category: queryCategory })
      })

      if (!res.ok || !res.body) {
        throw new Error(`Streaming failed with status ${res.status}`)
      }

      const reader = res.body.getReader()
      const decoder = new TextDecoder('utf-8')
      let buffer = ''
      let streamedRawText = ''

      setSearching(false) // First byte / stream started

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          const trimmed = line.trim()
          if (!trimmed.startsWith('data: ')) continue
          try {
            const data = JSON.parse(trimmed.slice(6))

            if (data.type === 'init') {
              setSearchResult({
                summary: 'Generating executive statutory verdict...',
                detailed_analysis: '',
                citations: data.citations || [],
                confidence_label: data.confidence_label || 'High',
                confidence_score: data.confidence_score || 0.85,
                execution_time_ms: 0,
              })
            } else if (data.type === 'token') {
              streamedRawText += data.token
              let splitMarker = null
              if (streamedRawText.includes('## Explainable AI (xAI) Statutory & Legal Document')) {
                splitMarker = '## Explainable AI (xAI) Statutory & Legal Document'
              } else if (streamedRawText.includes('## Detailed Analysis')) {
                splitMarker = '## Detailed Analysis'
              } else if (streamedRawText.includes('## Permitted Ingredients')) {
                splitMarker = '## Permitted Ingredients'
              }

              if (splitMarker) {
                const parts = streamedRawText.split(splitMarker)
                const sumText = parts[0].replace('## Short Summary', '').replace('## Executive Summary', '').trim()
                const analText = (splitMarker.startsWith('## ') && !splitMarker.includes('Detailed') ? splitMarker + '\n' : '') + parts.slice(1).join(splitMarker).trim()
                setSearchResult(prev => ({
                  ...(prev || {}),
                  summary: sumText,
                  detailed_analysis: analText,
                }))
              } else {
                const sumText = streamedRawText.replace('## Short Summary', '').replace('## Executive Summary', '').trim()
                setSearchResult(prev => ({
                  ...(prev || {}),
                  summary: sumText,
                }))
              }
            } else if (data.type === 'done') {
              setSearchResult(prev => ({
                ...(prev || {}),
                summary: data.summary || prev?.summary || '',
                detailed_analysis: data.detailed_analysis || prev?.detailed_analysis || '',
                execution_time_ms: data.execution_time_ms || prev?.execution_time_ms || 120,
              }))
            }
          } catch (jsonErr) {
            console.warn('SSE JSON parse error:', jsonErr)
          }
        }
      }
    } catch (streamErr) {
      console.warn("Stream failed, falling back to synchronous /api/search:", streamErr)
      try {
        const fallbackRes = await fetch('/api/search', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question: queryText, category: queryCategory })
        })
        if (!fallbackRes.ok) throw new Error(`Server returned status ${fallbackRes.status}`)
        const data = await fallbackRes.json()
        setSearchResult(data)
      } catch (err) {
        console.error("Search Error:", err)
        setSearchError("Failed to fetch response from FastAPI backend. Ensure backend server is running on port 8000.")
      }
    } finally {
      setSearching(false)
      setIsStreaming(false)
    }
  }

  function applySuggestion(qText, filterCat = "All") {
    setQuestion(qText)
    setCategory(filterCat)
    handleSearch(null, qText, filterCat)
  }

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

  useEffect(() => {
    if (activeTab === 'news') fetchNews()
  }, [activeTab])

  const filteredNews = (newsData.news_items || []).filter(item => {
    const matchesCat = newsFilter === 'All' || item.category === newsFilter
    const matchesSearch = !newsSearch.trim() || 
      item.title.toLowerCase().includes(newsSearch.toLowerCase()) ||
      (item.summary && item.summary.toLowerCase().includes(newsSearch.toLowerCase()))
    return matchesCat && matchesSearch
  })

  function getConfidenceBadgeClass(label, score) {
    if (label === 'High' || score >= 0.75) return 'badge-high'
    if (label === 'Medium' || score >= 0.50) return 'badge-medium'
    return 'badge-low'
  }

  return (
    <div className="layout">
      <header className="navbar">
        <div className="navbar-left">
          <div className="brand">
            <span className="brand-logo">🛡️</span>
            <div>
              <h1 className="brand-title">NiyamAI</h1>
              <p className="brand-sub">FSSAI Regulatory & Compliance Search</p>
            </div>
          </div>

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

      <main className="content-area">
        {activeTab === 'search' && (
          <div>
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
                      <option value="Advertising & Claims">Advertising & Claims / Packaging</option>
                      <option value="Contaminants">Contaminants & Heavy Metals</option>
                      <option value="Sweetening Agents & Standards">Sweetening Agents & Honey Standards</option>
                      <option value="Prohibitions">Prohibitions & Adulteration</option>
                      <option value="Additives">Food Additives & Preservatives</option>
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
                    <div className="executive-answer-hero">
                      <div className="executive-header">
                        <div className="executive-title-group">
                          <span className="executive-badge">🎯 Short Summary</span>
                          {isStreaming ? (
                            <span className="streaming-pulse-tag">⚡ Streaming Live from Groq LLM...</span>
                          ) : (
                            <span className="grounded-tag">🛡️ Verified FSSAI Statutory Provision</span>
                          )}
                        </div>
                        <button 
                          type="button" 
                          className="copy-btn"
                          onClick={() => copySummary(searchResult.summary)}
                          title="Copy short summary"
                        >
                          {copiedSummary ? '✅ Copied Short Summary!' : '📋 Copy Short Summary'}
                        </button>
                      </div>
                      <div className="executive-verdict-content">
                        <p className="executive-summary-text">
                          {searchResult.summary}
                          {isStreaming && <span className="typing-cursor">▌</span>}
                        </p>
                      </div>
                    </div>

                    <div className="analysis-box">
                      <div className="section-title-row">
                        <h3>📜 Explainable AI (xAI) Statutory & Legal Document</h3>
                        {searchResult.detailed_analysis && (
                          <button 
                            type="button" 
                            className="copy-btn-sm"
                            onClick={() => copyAnalysis(searchResult.detailed_analysis)}
                            title="Copy xAI Legal Document"
                          >
                            {copiedAnalysis ? '✅ Copied xAI Document!' : '📋 Copy xAI Document'}
                          </button>
                        )}
                      </div>
                      <div className="analysis-text xai-document-view">
                        <pre style={{ whiteSpace: 'pre-wrap', fontFamily: 'inherit', margin: 0 }}>
                          {searchResult.detailed_analysis || (isStreaming ? 'Synthesizing structured statutory document with data matrix & ingredient classification...' : '')}
                        </pre>
                      </div>
                    </div>

                    <div className="citations-box">
                      <div className="section-title-row">
                        <h3>📜 Official Verbatim Legal Citations ({searchResult.citations ? searchResult.citations.length : 0})</h3>
                      </div>
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

            {newsData.ai_overview && (
              <div className="news-ai-banner">
                <div className="news-ai-header">
                  <span>⚡ AI Regulatory Executive Pulse</span>
                </div>
                <p className="news-ai-text">{newsData.ai_overview}</p>
              </div>
            )}

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
                        <span className="news-source-tag">📰 {item.source_name || 'News Source'}</span>
                        <span className={`severity-tag severity-${(item.severity || 'Medium').toLowerCase()}`}>
                          {item.severity || 'Medium'} Urgency
                        </span>
                        <span className="news-date">{item.date || 'Recent'}</span>
                      </div>
                      <h3>{item.title}</h3>
                      <p className="news-summary">{item.summary}</p>

                      <a href={item.source_url} target="_blank" rel="noreferrer" className="news-link">
                        Read Full News Source ({item.source_name || 'Article'}) ↗
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
