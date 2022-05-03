import {
  AlertDialog,
  AlertDialogBody,
  AlertDialogContent,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogOverlay,
  Box,
  Button,
  Center,
  HStack,
  Text,
  useColorModeValue,
  VStack,
} from '@chakra-ui/react';
import { Split } from '@geoffcox/react-splitter';
import { useWindowSize } from '@react-hook/window-size';
import { app } from 'electron';
import log from 'electron-log';
import { memo, useEffect, useRef, useState } from 'react';
import { EdgeTypes, NodeTypes, ReactFlowProvider } from 'react-flow-renderer';
import useFetch, { CachePolicies } from 'use-http';
import { ipcRenderer } from '../helpers/safeIpc';
import ChaiNNerLogo from '../components/chaiNNerLogo';
import CustomEdge from '../components/CustomEdge';
import Header from '../components/Header';
import IteratorHelperNode from '../components/node/IteratorHelperNode';
import IteratorNode from '../components/node/IteratorNode';
import Node from '../components/node/Node';
import NodeSelector from '../components/NodeSelectorPanel';
import ReactFlowBox from '../components/ReactFlowBox';
import { GlobalProvider } from '../helpers/contexts/GlobalNodeState';
import { SettingsProvider } from '../helpers/contexts/SettingsContext';
import { BackendNodesResponse } from '../helpers/Backend';
import { SchemaMap } from '../common-types';

interface MainProps {
  port: number;
}

const Main = ({ port }: MainProps) => {
  // console.log('🚀 ~ file: main.jsx ~ line 27 ~ Main ~ port', port);
  const [availableNodes, setAvailableNodes] = useState<SchemaMap | null>(null);
  const [nodeTypes, setNodeTypes] = useState<NodeTypes | null>(null);
  const edgeTypes: EdgeTypes = {
    main: CustomEdge,
  };
  // const { colorMode, toggleColorMode } = useColorMode();
  const [, height] = useWindowSize();

  const reactFlowWrapper = useRef<HTMLDivElement>(null);

  // Queries
  const [backendReady, setBackendReady] = useState(false);

  const { loading, error, data, response } = useFetch<BackendNodesResponse>(
    `http://localhost:${port}/nodes`,
    { cachePolicy: CachePolicies.NO_CACHE, retries: 10 },
    [port]
  );

  const bgColor = useColorModeValue('gray.200', '#151a24');

  useEffect(() => {
    if (response.ok && data && !loading && !error && !backendReady) {
      setNodeTypes({
        regularNode: Node,
        iterator: IteratorNode,
        iteratorHelper: IteratorHelperNode,
      });
      const availableNodeMap: SchemaMap = {};
      data.forEach(({ category, nodes }) => {
        availableNodeMap[category] = {};
        nodes.forEach((node) => {
          availableNodeMap[category][node.name] = node;
        });
      });
      setAvailableNodes(availableNodeMap);
    }
  }, [response, data, loading, error, backendReady]);

  useEffect(() => {
    (async () => {
      if (nodeTypes && !backendReady) {
        setBackendReady(true);
        try {
          await ipcRenderer.invoke('backend-ready');
        } catch (err) {
          log.error(err);
        }
      }
    })();
  }, [nodeTypes]);

  const loadingLogo = (
    <ChaiNNerLogo
      percent={0}
      size={256}
    />
  );

  if (!nodeTypes || !availableNodes || !data) {
    return (
      <Box
        h="100vh"
        w="100vw"
      >
        <Center
          h="full"
          w="full"
        >
          <VStack>
            {loadingLogo}
            <Text>Loading...</Text>
          </VStack>
        </Center>
      </Box>
    );
  }

  if (error) {
    return (
      <AlertDialog
        isOpen
        onClose={() => {
          window.close();
          app.quit();
        }}
        // https://github.com/chakra-ui/chakra-ui/pull/5963
        leastDestructiveRef={undefined}
      >
        <AlertDialogOverlay>
          <AlertDialogContent>
            <AlertDialogHeader
              fontSize="lg"
              fontWeight="bold"
            >
              Critical Error
            </AlertDialogHeader>

            <AlertDialogBody>{error.message}</AlertDialogBody>

            <AlertDialogFooter>
              <Button
                colorScheme="red"
                ml={3}
                onClick={() => {
                  window.close();
                  app.quit();
                }}
              >
                Exit Application
              </Button>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialogOverlay>
      </AlertDialog>
    );
  }

  return (
    <ReactFlowProvider>
      <SettingsProvider port={port}>
        <GlobalProvider
          availableNodes={availableNodes}
          reactFlowWrapper={reactFlowWrapper}
        >
          <VStack
            bg={bgColor}
            overflow="hidden"
            p={2}
          >
            <Header port={port} />
            <HStack
              as={Split}
              defaultSplitterColors={{
                color: '#71809633',
                hover: '#71809666',
                drag: '#718096EE',
              }}
              initialPrimarySize="380px"
              minPrimarySize="290px"
              minSecondarySize="75%"
              splitterSize="10px"
            >
              <NodeSelector
                data={data}
                height={height}
              />

              <ReactFlowBox
                edgeTypes={edgeTypes}
                nodeTypes={nodeTypes}
                wrapperRef={reactFlowWrapper}
              />
            </HStack>
          </VStack>
        </GlobalProvider>
      </SettingsProvider>
    </ReactFlowProvider>
  );
};

export default memo(Main);