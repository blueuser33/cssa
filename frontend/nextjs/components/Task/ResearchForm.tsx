import React, { useState, useEffect } from "react";
import FileUpload from "../Settings/FileUpload";
import ToneSelector from "../Settings/ToneSelector";
import MCPSelector from "../Settings/MCPSelector";
import LayoutSelector from "../Settings/LayoutSelector";
import DomainFilter from "./DomainFilter";
import { useAnalytics } from "../../hooks/useAnalytics";
import { ChatBoxSettings, Domain, MCPConfig } from '@/types/data';

interface ResearchFormProps {
  chatBoxSettings: ChatBoxSettings;
  setChatBoxSettings: React.Dispatch<React.SetStateAction<ChatBoxSettings>>;
  onFormSubmit?: (
    task: string,
    reportType: string,
    reportSource: string,
    domains: Domain[]
  ) => void;
}

export default function ResearchForm({
  chatBoxSettings,
  setChatBoxSettings,
  onFormSubmit,
}: ResearchFormProps) {
  const { trackResearchQuery } = useAnalytics();
  const [task, setTask] = useState("");
  const [newDomain, setNewDomain] = useState('');

  // Destructure necessary fields from chatBoxSettings
  let { report_type, report_source, tone, layoutType } = chatBoxSettings;

  const [domains, setDomains] = useState<Domain[]>(() => {
    if (typeof window !== 'undefined') {
      const saved = localStorage.getItem('domainFilters');
      return saved ? JSON.parse(saved) : [];
    }
    return [];
  });
  
  useEffect(() => {
    localStorage.setItem('domainFilters', JSON.stringify(domains));
    setChatBoxSettings(prev => ({
      ...prev,
      domains: domains.map(domain => domain.value)
    }));
  }, [domains, setChatBoxSettings]);

  const handleAddDomain = (e: React.FormEvent) => {
    e.preventDefault();
    if (newDomain.trim()) {
      setDomains([...domains, { value: newDomain.trim() }]);
      setNewDomain('');
    }
  };

  const handleRemoveDomain = (domainToRemove: string) => {
    setDomains(domains.filter(domain => domain.value !== domainToRemove));
  };

  const onFormChange = (e: { target: { name: any; value: any } }) => {
    const { name, value } = e.target;
    setChatBoxSettings((prevSettings: any) => ({
      ...prevSettings,
      [name]: value,
    }));
  };

  const onToneChange = (e: { target: { value: any } }) => {
    const { value } = e.target;
    setChatBoxSettings((prevSettings: any) => ({
      ...prevSettings,
      tone: value,
    }));
  };

  const onLayoutChange = (e: { target: { value: any } }) => {
    const { value } = e.target;
    setChatBoxSettings((prevSettings: any) => ({
      ...prevSettings,
      layoutType: value,
    }));
  };

  const onMCPChange = (enabled: boolean, configs: MCPConfig[]) => {
    setChatBoxSettings((prevSettings: any) => ({
      ...prevSettings,
      mcp_enabled: enabled,
      mcp_configs: configs,
    }));
  };

  const onAcademicSourceChange = (source: string, checked: boolean) => {
    setChatBoxSettings((prevSettings: any) => {
      const currentSources = prevSettings.academic_sources || ["arxiv"];
      const nextSources = checked
        ? Array.from(new Set([...currentSources, source]))
        : currentSources.filter((item: string) => item !== source);

      return {
        ...prevSettings,
        academic_sources: nextSources.length > 0 ? nextSources : ["arxiv"],
      };
    });
  };

  const onAcademicNumberChange = (name: string, value: string) => {
    setChatBoxSettings((prevSettings: any) => ({
      ...prevSettings,
      [name]: value === "" ? null : Number(value),
    }));
  };

  const onAcademicBooleanChange = (name: string, checked: boolean) => {
    setChatBoxSettings((prevSettings: any) => ({
      ...prevSettings,
      [name]: checked,
    }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (onFormSubmit) {
      const updatedSettings = {
        ...chatBoxSettings,
        domains: domains.map(domain => domain.value)
      };
      setChatBoxSettings(updatedSettings);
      onFormSubmit(task, report_type, report_source, domains);
    }
  };

  return (
    <form
      method="POST"
      className="report_settings_static mt-3"
      onSubmit={handleSubmit}
    >
      <div className="form-group">
        <label htmlFor="report_type" className="agent_question">
          Report Type{" "}
        </label>
        <select
          name="report_type"
          value={report_type}
          onChange={onFormChange}
          className="form-control-static"
          required
        >
          <option value="research_report">
            Summary - Short and fast (~2 min)
          </option>
          <option value="deep">Deep Research Report</option>
          <option value="multi_agents">Multi Agents Report</option>
          <option value="detailed_report">
            Detailed - In depth and longer (~5 min)
          </option>
          <option value="academic_survey">
            Academic Survey - Papers, taxonomy, citation audit
          </option>
        </select>
      </div>

      {report_type === "academic_survey" ? (
        <div className="academic-settings">
          <div className="form-group">
            <label className="agent_question">Academic Sources</label>
            <div className="academic-source-row">
              <label>
                <input
                  type="checkbox"
                  checked={(chatBoxSettings.academic_sources || ["arxiv"]).includes("arxiv")}
                  onChange={(e) => onAcademicSourceChange("arxiv", e.target.checked)}
                />
                arXiv
              </label>
            </div>
          </div>

          <div className="form-group academic-grid">
            <label className="agent_question">Year Range</label>
            <div className="academic-range">
              <input
                className="input-static"
                type="number"
                min="1900"
                max="2100"
                value={chatBoxSettings.year_from ?? ""}
                onChange={(e) => onAcademicNumberChange("year_from", e.target.value)}
                placeholder="From"
              />
              <input
                className="input-static"
                type="number"
                min="1900"
                max="2100"
                value={chatBoxSettings.year_to ?? ""}
                onChange={(e) => onAcademicNumberChange("year_to", e.target.value)}
                placeholder="To"
              />
            </div>
          </div>

          <div className="form-group">
            <label htmlFor="max_papers" className="agent_question">
              Max Papers
            </label>
            <input
              id="max_papers"
              className="input-static"
              type="number"
              min="5"
              max="100"
              value={chatBoxSettings.max_papers ?? 30}
              onChange={(e) => onAcademicNumberChange("max_papers", e.target.value)}
            />
          </div>

          <div className="form-group">
            <label htmlFor="language" className="agent_question">
              Output Language
            </label>
            <select
              id="language"
              name="language"
              value={chatBoxSettings.language || "zh"}
              onChange={onFormChange}
              className="form-control-static"
            >
              <option value="zh">中文</option>
              <option value="english">English</option>
            </select>
          </div>

          <div className="form-group academic-checkbox">
            <label>
              <input
                type="checkbox"
                checked={chatBoxSettings.enable_citation_audit ?? true}
                onChange={(e) => onAcademicBooleanChange("enable_citation_audit", e.target.checked)}
              />
              Enable citation audit
            </label>
          </div>
        </div>
      ) : null}

      <div className="form-group">
        <label htmlFor="report_source" className="agent_question">
          Report Source{" "}
        </label>
        <select
          name="report_source"
          value={report_source}
          onChange={onFormChange}
          className="form-control-static"
          required
        >
          <option value="web">The Internet</option>
          <option value="local">My Documents</option>
          <option value="hybrid">Hybrid</option>
        </select>
      </div>

      

      {report_source === "local" || report_source === "hybrid" ? (
        <FileUpload />
      ) : null}
      
      <ToneSelector tone={tone} onToneChange={onToneChange} />

      <MCPSelector 
        mcpEnabled={chatBoxSettings.mcp_enabled || false}
        mcpConfigs={chatBoxSettings.mcp_configs || []}
        onMCPChange={onMCPChange}
      />
      
      <LayoutSelector layoutType={layoutType || 'copilot'} onLayoutChange={onLayoutChange} />

      {/** TODO: move the below to its own component */}
      {(chatBoxSettings.report_source === "web" || chatBoxSettings.report_source === "hybrid") && (
        <DomainFilter
          domains={domains}
          newDomain={newDomain}
          setNewDomain={setNewDomain}
          onAddDomain={handleAddDomain}
          onRemoveDomain={handleRemoveDomain}
        />
      )}
    </form>
  );
}
