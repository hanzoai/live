interface StatusBarProps {
  fps?: number;
  bitrate?: number;
}

const formatBitrate = (bps?: number): string => {
  if (bps === undefined || bps === 0) return "N/A";
  return bps >= 1000000
    ? `${(bps / 1000000).toFixed(1)} Mbps`
    : `${Math.round(bps / 1000)} kbps`;
};

const Metric = ({ label, value }: { label: string; value: string }) => (
  <div className="app-metric">
    <b>{label}:</b>
    <span>{value}</span>
  </div>
);

export function StatusBar({ fps, bitrate }: StatusBarProps) {
  return (
    <div className="app-statusbar">
      <Metric
        label="FPS"
        value={fps !== undefined && fps > 0 ? fps.toFixed(1) : "N/A"}
      />
      <Metric label="Bitrate" value={formatBitrate(bitrate)} />
    </div>
  );
}
