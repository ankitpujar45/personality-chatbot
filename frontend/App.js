import React, { useEffect, useRef, useState } from "react";
import "./App.css";

const API_BASE = "http://localhost:8000";
const USER_ID = "ankit";

function App() {
  const [personas, setPersonas] = useState({});
  const [selectedPersona, setSelectedPersona] = useState(null);
  const [sessionsByPersona, setSessionsByPersona] = useState({});
  const [activeConversationByPersona, setActiveConversationByPersona] = useState({});
  const [messageCache, setMessageCache] = useState({});
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isLoadingConversation, setIsLoadingConversation] = useState(false);
  const [isSending, setIsSending] = useState(false);

  const chatEndRef = useRef(null);

  const activeConversationId = selectedPersona ? activeConversationByPersona[selectedPersona] : null;
  const activeSessionList = selectedPersona ? sessionsByPersona[selectedPersona] || [] : [];
  const activePersonaMeta = selectedPersona ? personas[selectedPersona] : null;

  const conversationKey = (persona, conversationId) => `${persona}:${conversationId}`;

  const toUiMessages = (history) =>
    history.map((item) => ({
      sender: item.role === "assistant" ? "bot" : "user",
      text: item.content,
      emotion: item.detected_emotion,
      confidence: item.confidence
    }));

  const updateSessionList = (persona, updater) => {
    setSessionsByPersona((prev) => ({
      ...prev,
      [persona]: updater(prev[persona] || [])
    }));
  };

  const upsertSession = (persona, session) => {
    updateSessionList(persona, (current) => {
      const filtered = current.filter((item) => item.id !== session.id);
      return [session, ...filtered].sort((a, b) => (b.updated_at || "").localeCompare(a.updated_at || ""));
    });
  };

  const formatTimestamp = (value) => {
    if (!value) {
      return "";
    }

    const date = new Date(value);
    if (Number.isNaN(date.getTime())) {
      return "";
    }

    return date.toLocaleDateString([], {
      month: "short",
      day: "numeric"
    });
  };

  const loadConversation = async (persona, conversationId) => {
    if (!persona || !conversationId) {
      return;
    }

    setActiveConversationByPersona((prev) => ({
      ...prev,
      [persona]: conversationId
    }));

    const cached = messageCache[conversationKey(persona, conversationId)];
    if (cached) {
      setMessages(cached);
      return;
    }

    setIsLoadingConversation(true);

    try {
      const res = await fetch(
        `${API_BASE}/chat-history/${persona}/${conversationId}?user_id=${encodeURIComponent(USER_ID)}`
      );
      const data = await res.json();
      const loadedMessages = toUiMessages(data.messages || []);

      setMessageCache((prev) => ({
        ...prev,
        [conversationKey(persona, conversationId)]: loadedMessages
      }));
      setMessages(loadedMessages);

      if (data.session) {
        upsertSession(persona, data.session);
      }
    } catch (err) {
      console.error(err);
      setMessages([]);
    } finally {
      setIsLoadingConversation(false);
    }
  };

  const createNewChat = async (persona = selectedPersona) => {
    if (!persona) {
      return null;
    }

    try {
      const res = await fetch(`${API_BASE}/chat-sessions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          persona,
          user_id: USER_ID,
          title: "New chat"
        })
      });
      const data = await res.json();
      const session = data.session;

      if (!session) {
        return null;
      }

      upsertSession(persona, session);
      setActiveConversationByPersona((prev) => ({
        ...prev,
        [persona]: session.id
      }));
      setMessageCache((prev) => ({
        ...prev,
        [conversationKey(persona, session.id)]: []
      }));
      setMessages([]);
      setInput("");
      return session;
    } catch (err) {
      console.error(err);
      return null;
    }
  };

  const loadSessions = async (persona, preferredConversationId = null) => {
    if (!persona) {
      return;
    }

    try {
      const res = await fetch(
        `${API_BASE}/chat-sessions/${persona}?user_id=${encodeURIComponent(USER_ID)}`
      );
      const data = await res.json();
      const sessions = data.sessions || [];

      setSessionsByPersona((prev) => ({
        ...prev,
        [persona]: sessions
      }));

      let targetConversationId = preferredConversationId;
      if (!targetConversationId) {
        targetConversationId = activeConversationByPersona[persona] || sessions[0]?.id || null;
      }

      if (!targetConversationId) {
        await createNewChat(persona);
        return;
      }

      await loadConversation(persona, targetConversationId);
    } catch (err) {
      console.error(err);
    }
  };

  const switchPersona = async (persona) => {
    setSelectedPersona(persona);
    setInput("");
    setMessages([]);
    await loadSessions(persona);
  };

  const sendMessage = async () => {
    const message = input.trim();
    if (!message || !selectedPersona || isSending) {
      return;
    }

    let conversationId = activeConversationId;
    if (!conversationId) {
      const createdSession = await createNewChat(selectedPersona);
      conversationId = createdSession?.id || null;
    }

    if (!conversationId) {
      return;
    }

    const userMsg = { sender: "user", text: message };
    const cacheKey = conversationKey(selectedPersona, conversationId);

    setIsSending(true);
    setMessages((prev) => {
      const updated = [...prev, userMsg];
      setMessageCache((cache) => ({
        ...cache,
        [cacheKey]: updated
      }));
      return updated;
    });
    setInput("");

    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message,
          persona: selectedPersona,
          user_id: USER_ID,
          conversation_id: conversationId
        })
      });

      if (!res.ok) {
        throw new Error(`Chat request failed with status ${res.status}`);
      }

      const data = await res.json();
      const botMsg = {
        sender: "bot",
        text:
          typeof data.response === "string" && data.response.trim()
            ? data.response.trim()
            : "Something went wrong on the reply."
      };

      setMessages((prev) => {
        const updated = [...prev, botMsg];
        setMessageCache((cache) => ({
          ...cache,
          [cacheKey]: updated
        }));
        return updated;
      });

      if (data.session) {
        upsertSession(selectedPersona, data.session);
        setActiveConversationByPersona((prev) => ({
          ...prev,
          [selectedPersona]: data.session.id
        }));
      }
    } catch (err) {
      console.error(err);

      setMessages((prev) => {
        const updated = [...prev, { sender: "bot", text: "server error" }];
        setMessageCache((cache) => ({
          ...cache,
          [cacheKey]: updated
        }));
        return updated;
      });
    } finally {
      setIsSending(false);
    }
  };

  useEffect(() => {
    fetch(`${API_BASE}/personas`)
      .then((res) => res.json())
      .then(async (data) => {
        setPersonas(data);
        const firstPersona = Object.keys(data)[0] || null;
        if (firstPersona) {
          setSelectedPersona(firstPersona);
          try {
            const sessionsRes = await fetch(
              `${API_BASE}/chat-sessions/${firstPersona}?user_id=${encodeURIComponent(USER_ID)}`
            );
            const sessionsData = await sessionsRes.json();
            const sessions = sessionsData.sessions || [];

            setSessionsByPersona((prev) => ({
              ...prev,
              [firstPersona]: sessions
            }));

            if (sessions.length > 0) {
              const firstConversationId = sessions[0].id;
              setActiveConversationByPersona((prev) => ({
                ...prev,
                [firstPersona]: firstConversationId
              }));

              const historyRes = await fetch(
                `${API_BASE}/chat-history/${firstPersona}/${firstConversationId}?user_id=${encodeURIComponent(USER_ID)}`
              );
              const historyData = await historyRes.json();
              const loadedMessages = toUiMessages(historyData.messages || []);

              setMessageCache((prev) => ({
                ...prev,
                [conversationKey(firstPersona, firstConversationId)]: loadedMessages
              }));
              setMessages(loadedMessages);
            } else {
              const createRes = await fetch(`${API_BASE}/chat-sessions`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                  persona: firstPersona,
                  user_id: USER_ID,
                  title: "New chat"
                })
              });
              const createData = await createRes.json();
              const newSession = createData.session;

              if (newSession) {
                setSessionsByPersona((prev) => ({
                  ...prev,
                  [firstPersona]: [newSession]
                }));
                setActiveConversationByPersona((prev) => ({
                  ...prev,
                  [firstPersona]: newSession.id
                }));
                setMessageCache((prev) => ({
                  ...prev,
                  [conversationKey(firstPersona, newSession.id)]: []
                }));
                setMessages([]);
              }
            }
          } catch (err) {
            console.error(err);
          }
        }
      });
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <div className="appShell">
      <aside className="sidebar">
        <div className="sidebarTop">
          <div className="brandBlock">
            <p className="brandEyebrow">Personality Chatbot</p>
            <h1>Chats</h1>
          </div>

          <button
            className="newChatButton"
            onClick={() => createNewChat()}
            disabled={!selectedPersona}
          >
            + New Chat
          </button>
        </div>

        <div className="sidebarSection">
          <p className="sectionLabel">Personas</p>
          <div className="personaList">
            {Object.entries(personas).map(([key, persona]) => (
              <button
                key={key}
                className={`personaButton ${selectedPersona === key ? "active" : ""}`}
                onClick={() => switchPersona(key)}
              >
                <span className="personaButtonName">{persona.name}</span>
                <span className="personaButtonTag">{persona.tagline}</span>
              </button>
            ))}
          </div>
        </div>

        <div className="sidebarSection historySection">
          <p className="sectionLabel">History</p>
          <div className="historyList">
            {activeSessionList.length === 0 ? (
              <div className="historyEmpty">Start a chat and it will show up here.</div>
            ) : (
              activeSessionList.map((session) => (
                <button
                  key={session.id}
                  className={`historyItem ${activeConversationId === session.id ? "active" : ""}`}
                  onClick={() => loadConversation(selectedPersona, session.id)}
                >
                  <span className="historyTitle">{session.title || "New chat"}</span>
                  <span className="historyPreview">{session.last_message_preview || "No messages yet"}</span>
                  <span className="historyMeta">{formatTimestamp(session.updated_at)}</span>
                </button>
              ))
            )}
          </div>
        </div>
      </aside>

      <main className="mainPanel">
        <div className="chatHeader">
          <div>
            <h2>{activePersonaMeta?.name || "Choose a persona"}</h2>
            <p>{activePersonaMeta?.tagline || "Persona-driven conversation"}</p>
          </div>
        </div>

        <div className="chatbox">
          {isLoadingConversation ? (
            <div className="emptyState">
              <h3>Loading chat...</h3>
            </div>
          ) : messages.length === 0 ? (
            <div className="emptyState">
              <h3>{activePersonaMeta?.name || "New chat"}</h3>
              <p>
                {selectedPersona === "ankit" && "Talk casually, rant, joke around, or vent like you would with a real friend."}
                {selectedPersona === "mentor" && "Drop your problem and get a direct, no-fluff response."}
                {selectedPersona === "therapist" && "Start wherever you are emotionally and let the conversation unfold slowly."}
              </p>
            </div>
          ) : (
            messages.map((message, index) => (
              <div key={index} className={`bubble ${message.sender}`}>
                {message.text}
              </div>
            ))
          )}
          <div ref={chatEndRef}></div>
        </div>

        <div className="composer">
          <div className="composerInner">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={
                selectedPersona === "ankit"
                  ? "Message your chill friend..."
                  : selectedPersona === "mentor"
                    ? "Say what happened..."
                    : "Share what is on your mind..."
              }
              onKeyDown={(e) => e.key === "Enter" && sendMessage()}
            />
            <button onClick={sendMessage} disabled={!selectedPersona || isSending}>
              {isSending ? "Sending..." : "Send"}
            </button>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
