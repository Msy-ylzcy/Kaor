import {
  Baseline,
  Check,
  FileOutput,
  FileVideo2,
  Languages,
  ScanLine,
  ScanText,
  Table2,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import type { WorkflowStepId } from "../types";

const steps: Array<{
  id: WorkflowStepId;
  label: string;
  icon: LucideIcon;
}> = [
  { id: "import", label: "导入", icon: FileVideo2 },
  { id: "roi", label: "框选", icon: ScanLine },
  { id: "ocr", label: "识别", icon: ScanText },
  { id: "csv", label: "校对", icon: Table2 },
  { id: "translate", label: "翻译", icon: Languages },
  { id: "layout", label: "排版", icon: Baseline },
  { id: "export", label: "导出", icon: FileOutput },
];

interface StepRailProps {
  active: WorkflowStepId;
  completed: Set<WorkflowStepId>;
  onSelect: (step: WorkflowStepId) => void;
}

export function StepRail({ active, completed, onSelect }: StepRailProps) {
  return (
    <nav className="step-rail" aria-label="处理流程">
      {steps.map((step, index) => {
        const Icon = step.icon;
        const isActive = step.id === active;
        const isComplete = completed.has(step.id);
        return (
          <button
            key={step.id}
            type="button"
            className={`step-button ${isActive ? "is-active" : ""} ${
              isComplete ? "is-complete" : ""
            }`}
            onClick={() => onSelect(step.id)}
            aria-current={isActive ? "step" : undefined}
            title={`${index + 1}. ${step.label}`}
          >
            <span className="step-icon">
              <Icon size={19} strokeWidth={1.9} />
              {isComplete && <Check className="step-check" size={10} strokeWidth={3} />}
            </span>
            <span>{step.label}</span>
          </button>
        );
      })}
    </nav>
  );
}
