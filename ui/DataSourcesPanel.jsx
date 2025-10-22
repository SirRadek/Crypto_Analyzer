import React, { useMemo, useState } from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  CardHeader,
  CircularProgress,
  Divider,
  FormControlLabel,
  Grid,
  Stack,
  Switch,
  TextField,
  Typography,
} from '@mui/material';
import { useTheme } from '@mui/material/styles';

const DEFAULT_SOURCES = [
  {
    id: 'reddit',
    label: 'Reddit',
    enabled: true,
    intervalMinutes: 5,
    lastUpdated: '10 min ago',
    records: 865,
  },
  {
    id: 'twitter',
    label: 'Twitter',
    enabled: true,
    intervalMinutes: 2,
    lastUpdated: 'Just now',
    records: 1320,
  },
  {
    id: 'whaleAlert',
    label: 'Whale Alert',
    enabled: true,
    intervalMinutes: 3,
    lastUpdated: '5 min ago',
    records: 342,
  },
  {
    id: 'binance',
    label: 'Binance',
    enabled: false,
    intervalMinutes: 8,
    lastUpdated: '22 min ago',
    records: 714,
  },
  {
    id: 'glassnode',
    label: 'Glassnode',
    enabled: true,
    intervalMinutes: 15,
    lastUpdated: '1 hr ago',
    records: 129,
  },
];

function formatRecords(records) {
  if (records === null || records === undefined) {
    return '—';
  }

  return Number.isFinite(records) ? records.toLocaleString() : records;
}

function DataSourcesPanel({
  initialSources = DEFAULT_SOURCES,
  handleSettingsChange,
  handleManualFetch,
  isLoading = false,
}) {
  const theme = useTheme();
  const [sources, setSources] = useState(() => initialSources);

  const enabledCount = useMemo(
    () => sources.filter((source) => source.enabled).length,
    [sources],
  );

  const updateSource = (sourceId, changes) => {
    setSources((prevSources) =>
      prevSources.map((source) =>
        source.id === sourceId ? { ...source, ...changes } : source,
      ),
    );

    if (typeof handleSettingsChange === 'function') {
      const existingSource = sources.find((source) => source.id === sourceId);
      const nextEnabled = Object.prototype.hasOwnProperty.call(
        changes,
        'enabled',
      )
        ? changes.enabled
        : existingSource?.enabled;
      const nextInterval = Object.prototype.hasOwnProperty.call(
        changes,
        'intervalMinutes',
      )
        ? changes.intervalMinutes
        : existingSource?.intervalMinutes;

      if (
        existingSource &&
        existingSource.enabled === nextEnabled &&
        existingSource.intervalMinutes === nextInterval
      ) {
        return;
      }

      if (nextEnabled !== undefined || nextInterval !== undefined) {
        handleSettingsChange(sourceId, {
          enabled: nextEnabled,
          intervalMinutes: nextInterval,
        });
      }
    }
  };

  const handleToggle = (sourceId) => (event) => {
    updateSource(sourceId, { enabled: event.target.checked });
  };

  const handleIntervalChange = (sourceId) => (event) => {
    const value = Number.parseInt(event.target.value, 10);
    const interval = Number.isNaN(value) ? 1 : Math.max(1, value);
    updateSource(sourceId, { intervalMinutes: interval });
  };

  const renderStatusValue = (value) => (
    <Typography variant="body2" fontWeight={600}>
      {isLoading ? <CircularProgress size={16} thickness={5} /> : value}
    </Typography>
  );

  return (
    <Box display="flex" flexDirection="column" gap={4}>
      <Box>
        <Typography variant="h4" gutterBottom>
          Data Source Management
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Toggle streaming sources, tune fetch intervals, and monitor live status at a glance.
        </Typography>
      </Box>

      <Grid container spacing={3}>
        {sources.map((source) => (
          <Grid item xs={12} sm={6} lg={4} key={source.id}>
            <Card
              variant="outlined"
              sx={{
                height: '100%',
                display: 'flex',
                flexDirection: 'column',
                backgroundColor:
                  theme.palette.mode === 'dark'
                    ? theme.palette.background.paper
                    : theme.palette.background.default,
              }}
            >
              <CardHeader
                title={
                  <Stack direction="row" spacing={1} alignItems="center">
                    <Box
                      component="span"
                      sx={{
                        width: 8,
                        height: 32,
                        borderRadius: 999,
                        backgroundColor: source.enabled
                          ? theme.palette.success.main
                          : theme.palette.grey[400],
                      }}
                    />
                    <Box>
                      <Typography variant="h6">{source.label}</Typography>
                      <Typography variant="caption" color="text.secondary">
                        Fetch every {source.intervalMinutes} minute(s)
                      </Typography>
                    </Box>
                  </Stack>
                }
                action={
                  <FormControlLabel
                    labelPlacement="start"
                    control={
                      <Switch
                        color="success"
                        checked={source.enabled}
                        onChange={handleToggle(source.id)}
                        disabled={isLoading}
                        inputProps={{ 'aria-label': `${source.label} toggle` }}
                      />
                    }
                    label={source.enabled ? 'Enabled' : 'Disabled'}
                    sx={{ m: 0 }}
                  />
                }
              />

              <Divider />

              <CardContent sx={{ flexGrow: 1 }}>
                <Stack spacing={2}>
                  <TextField
                    label="Interval (min)"
                    type="number"
                    value={source.intervalMinutes}
                    onChange={handleIntervalChange(source.id)}
                    inputProps={{ min: 1 }}
                    helperText="Minimum value is 1 minute"
                    fullWidth
                    disabled={isLoading}
                  />

                  <Stack
                    direction={{ xs: 'column', sm: 'row' }}
                    justifyContent="space-between"
                    alignItems={{ xs: 'flex-start', sm: 'center' }}
                    spacing={1}
                  >
                    <Typography variant="body2" color="text.secondary">
                      Last update
                    </Typography>
                    {renderStatusValue(source.lastUpdated)}
                  </Stack>

                  <Stack
                    direction={{ xs: 'column', sm: 'row' }}
                    justifyContent="space-between"
                    alignItems={{ xs: 'flex-start', sm: 'center' }}
                    spacing={1}
                  >
                    <Typography variant="body2" color="text.secondary">
                      Records fetched
                    </Typography>
                    {renderStatusValue(formatRecords(source.records))}
                  </Stack>

                  {typeof handleManualFetch === 'function' && (
                    <Box textAlign="right">
                      <Button
                        size="small"
                        variant="outlined"
                        onClick={() => handleManualFetch(source.id)}
                        disabled={isLoading}
                      >
                        Refresh Now
                      </Button>
                    </Box>
                  )}
                </Stack>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      <Box display="flex" justifyContent="flex-end">
        <Typography variant="caption" color="text.secondary">
          {enabledCount} of {sources.length} sources enabled
        </Typography>
      </Box>
    </Box>
  );
}

export default DataSourcesPanel;
